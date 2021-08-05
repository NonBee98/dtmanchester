import argparse
import datetime as dt
import json
import logging as default_logging
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pywt
import seglearn as sgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from matplotlib.collections import LineCollection
from pyts import datasets
from pyts.approximation import (PiecewiseAggregateApproximation,
                                SymbolicAggregateApproximation)
from pyts.preprocessing import *
from scipy import spatial
from scipy.stats import norm
from seglearn.feature_functions import *
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, FeatureRepMix, Segment
from sklearn import cluster, covariance, manifold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import *
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch import nn, optim
from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial
# Cell
from torch.distributions.geometric import Geometric
from tsai import *
from tsai.all import *
from tsai.data.core import *
from tsai.models.layers import *
from tsai.models.utils import *
from tsai.utils import *
from tslearn import metrics, piecewise, preprocessing
from tslearn.barycenters import (dtw_barycenter_averaging,
                                 dtw_barycenter_averaging_subgradient,
                                 euclidean_barycenter, softdtw_barycenter)
from tslearn.clustering import KShape, silhouette_score
from tslearn.preprocessing import (TimeSeriesResampler,
                                   TimeSeriesScalerMeanVariance)

INFINITY = float('inf')


def load_data(year):
    train_folder = './data/train/'
    # test_folder = './data/test/'
    stocklist = json.load(open(train_folder+'stockNames.json'))
    folder = train_folder+str(year)+'/'
    datas = {}
    dates = {}
    for stock in stocklist:
        file = folder+stock+'.csv'
        data = pd.read_csv(file)
        data, date = splite_data(data, -1)
        datas[stock] = data
        dates[stock] = date
    return stocklist, datas, dates


def pre_process_data(datas, seq_len=None, denosing=True, normalization=True, dimension=[1], compression='pip'):
    processed_data = []
    if seq_len is None:
        shortest = float('inf')
        for v in datas.values():
            if len(v) < shortest:
                shortest = len(v)
        seq_len = shortest

    downsample = preprocessing.TimeSeriesResampler(seq_len)
    ts_paa = piecewise.PiecewiseAggregateApproximation(n_segments=seq_len)

    for v in datas.values():
        data = v[:, 1]
        if denosing:
            data = wavelet_denoising(data)
        data = data[:, np.newaxis]
        if compression == 'pip':
            data = data[pip(data[:, 0], seq_len)]
        elif compression == 'paa':
            data = ts_paa.fit_transform(data[np.newaxis, :]).squeeze()
        else:
            data = downsample.fit_transform(data[np.newaxis, :]).squeeze()
        if normalization:
            data = zero_mean_normalise(data)
        processed_data.append(data.ravel())
    processed_data = np.array(processed_data)
    return processed_data


def getNames(nums, indexes):
    ret = []
    for i in range(len(indexes)):
        if indexes[i]:
            ret.append(nums[i])
    return ret


def get_fake_sample(data):
    sample_nums = data.shape[0]
    series_len = data.shape[1]
    mask = np.ones(shape=[sample_nums, series_len])
    rand_list = np.zeros(shape=[sample_nums, series_len])

    fake_position_nums = int(series_len * 0.2)
    fake_position = np.random.randint(low=0, high=series_len, size=[
                                      sample_nums, fake_position_nums])

    for i in range(fake_position.shape[0]):
        for j in range(fake_position.shape[1]):
            mask[i, fake_position[i, j]] = 0

    for i in range(rand_list.shape[0]):
        count = 0
        for j in range(rand_list.shape[1]):
            if j in fake_position[i]:
                rand_list[i, j] = data[i, fake_position[i, count]]
                count += 1
    fake_data = data * mask + rand_list * (1 - mask)
    real_fake_labels = np.zeros(shape=[sample_nums * 2, 2])
    for i in range(sample_nums * 2):
        if i < sample_nums:
            real_fake_labels[i, 0] = 1
        else:
            real_fake_labels[i, 1] = 1
    return fake_data, real_fake_labels

class PipItem(object):

    def __init__(self, index, parent, **kwargs):
        self.index = index
        self.parent = parent
        self.cache = kwargs.get('cache', INFINITY)
        self.value = kwargs.get('value', None)
        self.order = kwargs.get('order', None)
        self.left = kwargs.get('left', None)
        self.right = kwargs.get('right', None)

    def update_cache(self):
        if None in (self.left, self.right):
            self.cache = INFINITY
        else:
            self.cache = self.parent.distance(
                self.left.value, self.value, self.right.value)
        self.parent.notify_change(self.index)

    def put_after(self, tail):
        if tail is not None:
            tail.right = self
            tail.update_cache()
        self.left = tail
        self.update_cache()
        return self

    def recycle(self):
        if self.left is None:
            self.parent.head = self.right
        else:
            self.left.right = self.right
            self.left.update_cache()

        if self.right is None:
            self.parent.tail = self.left
        else:
            self.right.left = self.left
            self.right.update_cache()

        return self.clear()

    def clear(self):
        self.order = 0
        self.left = None
        self.right = None
        self.cache = INFINITY

        ret = self.value
        self.value = None
        return ret


class PipHeap(object):

    def __init__(self, distance, **kwargs):
        self.distance = distance
        self.heap = self.create_heap(512)
        self.head = None
        self.tail = None
        self.size = 0
        self.global_order = 0

    def create_heap(self, size):
        return [PipItem(i, self) for i in range(size)]

    def ensure_heap(self, size):
        new_elements = [PipItem(i, self)
                        for i in range(len(self.heap), size+1)]
        self.heap.extend(new_elements)

    def acquire_item(self, value):
        self.ensure_heap(self.size)
        item = self.heap[self.size]
        item.value = value

        self.size += 1
        self.global_order += 1
        item.order = self.global_order
        return item

    def add(self, value):
        self.tail = self.acquire_item(value).put_after(self.tail)
        if self.head is None:
            self.head = self.tail

    @property
    def min_value(self):
        return self.heap[0].cache

    def remove_min(self):
        return self.remove_at(0)

    def remove_at(self, index):
        self.size -= 1
        self.swap(index, self.size)
        self.bubble_down(index)
        return self.heap[self.size].recycle()

    def notify_change(self, index):
        return self.bubble_down(self.bubble_up(index))

    def bubble_up(self, n):
        while (n != 0) and self.less(n, (n-1)/2):
            n = self.swap(n, (n-1)/2)
        return n

    def bubble_down(self, n):
        def get_k(n): return self.min(n, n*2+1, n*2+2)

        k = get_k(n)
        while (k != n) and (k < self.size):
            n = self.swap(n, k)
            k = get_k(n)
        return n

    def min(self, i, j, k=None):
        if k is not None:
            result = self.min(i, self.min(j, k))
        else:
            result = i if self.less(i, j) else j

        return result

    def less(self, i, j):
        def i_smaller_than_j(heap, i, j):
            i, j = int(i), int(j)
            if heap[i].cache != heap[j].cache:
                result = heap[i].cache < heap[j].cache
            else:
                result = heap[i].order < heap[j].order
            return result

        heap = self.heap
        return ((i < self.size) and (j >= self.size or i_smaller_than_j(heap, i, j)))

    def swap(self, i, j):
        i, j = int(i), int(j)
        self.heap[i].index, self.heap[j].index = j, i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        return j

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.value[0]
            current = current.right


def vertical_distance(left, current, right):
    EPSILON = 1e-06
    a_x, a_y = left
    b_x, b_y = current
    c_x, c_y = right

    if (abs(a_x - b_x) < EPSILON) or (abs(b_x - c_x) < EPSILON):
        result = 0
    elif (c_x - a_x) == 0:
        # Otherwise we could have a ZeroDivisionError
        result = INFINITY
    else:
        result = abs(
            ((a_y + (c_y - a_y) * (b_x - a_x) / (c_x - a_x) - b_y)) * (c_x - a_x))

    return result


def dist(x1, x2, y1, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def euclidean_distance(left, current, right):
    left_current = dist(left[0], current[0], left[1], current[1])
    rightcurrent = dist(right[0], current[0], right[1], current[1])
    return (left_current + rightcurrent) * (right[0] - left[0])


def pip(data, k, fast=True, stream_mode=True, distance='vertical'):
    distance_functions = {
        'vertical': vertical_distance,
        'euclidean': euclidean_distance,
    }
    distance_function = distance_functions[distance]

    if fast:
        result = fastpip(data, k, stream_mode=stream_mode,
                         distance_function=distance_function)
    else:
        result = simplepip(data, k, distance_function=distance_function)

    return result


def fastpip(data, k, stream_mode=True, distance_function=vertical_distance):

    if len(data) >= k:
        heap = PipHeap(distance_function)

        for idx, element in enumerate(data):
            heap.add((idx, element))

            if stream_mode and (heap.size > k):
                heap.remove_min()

        if not stream_mode:
            while heap.size > k:
                heap.remove_min()

        ret = list(heap)
    else:
        ret = data

    return ret


def simplepip(data, k, distance_function=vertical_distance):
    ret = []

    for (idx, value) in enumerate(data):
        ret.append(value)
        if len(ret) <= k:
            continue

        miniv = sys.maxsize
        minij = 0

        for j in range(1, len(ret) - 1):
            d = distance_function(ret[j - 1], ret[j], ret[j + 1])
            if d < miniv:
                miniv = d
                minij = j

        del ret[minij]

    return ret


def splite_data(data, width=360, overlap=0.):
    '''splite time series data
    data (pdFrame): pandas table
    interval (int): the interval of each splite, expressed in days
    '''
    dates = data[['Date']].to_numpy()
    data = data[['Open','Close','High','Low']].to_numpy()
    if width <=0:
        return data, dates
    segment = Segment(width=width, overlap=overlap)
    data_segment, _, _ = segment.fit_transform([data], None)
    date_segment, _, _ = segment.fit_transform([dates], None)
    return data_segment, date_segment


def portion_normalise(data):
    '''
    Portion normalise a single segment
    data should have multiple dimension
    '''
    t, s = data.shape
    res = np.zeros((t-2, s))
    for i in range(s):
        for j in range(2, t):
            res[j-2][i] = (data[j][i] - data[j-1][i]) / \
                (data[j-1][i] - data[j-2][i] + 1e-6)
    return res


def min_max_normalise(data):
    '''
    Min_max normalise a single segment
    data should have multiple dimension
    '''
    if data.ndim == 1:
        mi = data.min()
        ma = data.max()
    else:
        mi = data.min(axis=0)
        ma = data.max(axis=0)
    res = (data-mi) / (ma-mi)
    return res


def zero_mean_normalise(data):
    '''
    z normalise a single segment
    data should have multiple dimension
    '''
    if data.ndim == 1:
        mean = data.mean()
        std = data.std()
    else:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
    res = (data-mean) / std
    return res


def wavelet_denoising(data):
    '''
    Denoising data 
    '''
    # db4 = pywt.Wavelet('db8')
    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(data, db4)
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], 0.04)
    meta = pywt.waverec(coeffs, db4)
    return meta


def f_ratio_euclidean(X, lb):
    """
    # Compute the f-ratio = k * ssw / ssb
    
    Input:
        - X: (n,d), n datapoints each with d dimension
        - lb: (n,) label of each datapoint, each element is an
              integer, >=0, <n.
    
    Return:
        - f_ratio = k * ssw / ssb: scalar
    """
    k = len(np.unique(lb))
    _, d = np.shape(X)
    n = np.zeros(k)
    c = np.zeros([k, d])
    ###
    labels = np.unique(lb)
    new_centers = []
    clusters = []
    ## find the data points of each cluster and get the center point
    global_mean = X.mean(axis=0)
    for label in labels:
        cluster = X[lb == label]
        clusters.append(cluster)
        new_centers.append(cluster.mean(axis=0))
    new_centers = np.array(new_centers)
    SSW = 0
    SSB = 0
    for i in range(len(new_centers)):
        cluster = np.array(clusters[i])
        # get ssw of each cluster
        tmp = cluster - new_centers[i]
        tmp = (tmp @ tmp.T).diagonal()
        t_ssw = np.sum(tmp)
        SSW += t_ssw
        # get ssb of each cluster
        tmp = global_mean - new_centers[i]
        t_ssb = len(clusters[i]) * tmp @ tmp.T
        SSB += t_ssb
    f_ratio = k * SSW / SSB

    return f_ratio

# mask
def create_subsequence_mask(o, r=.15, lm=3, stateful=True, sync=False):
    device = o.device
    if o.ndim == 2:
        o = o[None]
    n_masks, mask_dims, mask_len = o.shape
    if sync == 'random':
        sync = random.random() > .5
    dims = 1 if sync else mask_dims
    if stateful:
        numels = n_masks * dims * mask_len
        pm = torch.tensor([1 / lm], device=device)
        pu = torch.clip(pm * (r / max(1e-6, 1 - r)), 1e-3, 1)
        zot, proba_a, proba_b = (torch.as_tensor([False, True], device=device), pu, pm) if random.random() > pm else \
            (torch.as_tensor([True, False], device=device), pm, pu)
        max_len = max(1, 2 * math.ceil(numels // (1/pm + 1/pu)))
        for i in range(10):
            _dist_a = (Geometric(probs=proba_a).sample([max_len])+1).long()
            _dist_b = (Geometric(probs=proba_b).sample([max_len])+1).long()
            dist_a = _dist_a if i == 0 else torch.cat((dist_a, _dist_a), dim=0)
            dist_b = _dist_b if i == 0 else torch.cat((dist_b, _dist_b), dim=0)
            add = torch.add(dist_a, dist_b)
            if torch.gt(torch.sum(add), numels):
                break
        dist_len = torch.argmax((torch.cumsum(add, 0) >= numels).float()) + 1
        if dist_len % 2:
            dist_len += 1
        repeats = torch.cat(
            (dist_a[:dist_len], dist_b[:dist_len]), -1).flatten()
        zot = zot.repeat(dist_len)
        mask = torch.repeat_interleave(zot, repeats)[
            :numels].reshape(n_masks, dims, mask_len)
    else:
        probs = torch.tensor(r, device=device)
        mask = Binomial(1, probs).sample((n_masks, dims, mask_len)).bool()
    if sync:
        mask = mask.repeat(1, mask_dims, 1)
    return mask


def create_variable_mask(o, r=.15):
    device = o.device
    n_masks, mask_dims, mask_len = o.shape
    _mask = torch.zeros((n_masks * mask_dims, mask_len), device=device)
    if int(mask_dims * r) > 0:
        n_masked_vars = int(n_masks * mask_dims * r)
        p = torch.tensor([1./(n_masks * mask_dims)],
                         device=device).repeat([n_masks * mask_dims])
        sel_dims = p.multinomial(num_samples=n_masked_vars, replacement=False)
        _mask[sel_dims] = 1
    mask = _mask.reshape(*o.shape).bool()
    return mask


def create_future_mask(o, r=.15, sync=False):
    if o.ndim == 2:
        o = o[None]
    n_masks, mask_dims, mask_len = o.shape
    if sync == 'random':
        sync = random.random() > .5
    dims = 1 if sync else mask_dims
    probs = torch.tensor(r, device=o.device)
    mask = Binomial(1, probs).sample((n_masks, dims, mask_len))
    if sync:
        mask = mask.repeat(1, mask_dims, 1)
    mask = torch.sort(mask, dim=-1, descending=True)[0].bool()
    return mask


def natural_mask(o):
    """Applies natural missingness in a batch to non-nan values in the next sample"""
    mask1 = torch.isnan(o)
    mask2 = rotate_axis0(mask1)
    return torch.logical_and(mask2, ~mask1)

# Cell


def create_mask(o,  r=.15, lm=3, stateful=True, sync=False, subsequence_mask=True, variable_mask=False, future_mask=False):
    if r <= 0 or r >= 1:
        return torch.ones_like(o)
    if int(r * o.shape[1]) == 0:
        variable_mask = False
    if subsequence_mask and variable_mask:
        random_thr = 1/3 if sync == 'random' else 1/2
        if random.random() > random_thr:
            variable_mask = False
        else:
            subsequence_mask = False
    elif future_mask:
        return create_future_mask(o, r=r)
    elif subsequence_mask:
        return create_subsequence_mask(o, r=r, lm=lm, stateful=stateful, sync=sync)
    elif variable_mask:
        return create_variable_mask(o, r=r)
    else:
        raise ValueError(
            'You need to set subsequence_mask, variable_mask or future_mask to True or pass a custom mask.')
