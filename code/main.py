import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from pyts.approximation import PiecewiseAggregateApproximation, SymbolicAggregateApproximation
import seglearn as sgl
from seglearn.pipe import Pype
from seglearn.transform import Segment, FeatureRep, FeatureRepMix
from .utils import *
from seglearn.feature_functions import *
import json
import os
import torch
import datetime as dt
from tslearn.clustering import kshape, silhouette_score
from collections import Counter
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import pywt
from tslearn import metrics, preprocessing, piecewise
from scipy import spatial
from pyts.transformation import BagOfPatterns, ROCKET
from scipy.stats import norm
import matplotlib.lines as mlines
import random
from torch import nn
import iisignature


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


train_folder = './data/train/'
# test_folder = './data/test/'
stocklist = json.load(open(train_folder+'stockNames.json'))
for year in range(2011,2016):
    folder = train_folder+str(year)+'/'
    datas = {}
    dates = {}
    for stock in stocklist:
        file = folder+stock+'.csv'
        data = pd.read_csv(file)
        data, date = splite_data(data, -1)
        datas[stock] = data
        dates[stock] = date
