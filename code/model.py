import argparse
import datetime as dt
import json
import os
import pickle
import random
import time
from collections import Counter

import iisignature
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
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
from pyts.transformation import ROCKET, BagOfPatterns
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
from sklearn_som.som import SOM
from torch import nn, optim
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

from utils import *

# Useful function for arguments.




# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def batch_iter(x, batch_size=50, shuffle_data=True):
    """
    return batches for supervised model
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    x_shuffle = x
    if shuffle_data:
        x_shuffle = shuffle(
            x_shuffle)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        data = x_shuffle[start_id:end_id]
        # fake_data, real_fake_labels = get_fake_sample(x)
        # yield torch.Tensor(data,device=device), torch.Tensor(fake_data,device=device), torch.LongTensor(real_fake_labels,device=device)
        yield torch.Tensor(data).to(device)

class DCTR(nn.Module):

    def __init__(self, input_size, encoder_dim=50, class_num=4, batch_size=50, lamda=1, classification_loss=False):
        super(DCTR, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.encoder_dim = encoder_dim
        self.class_num = class_num
        self.K = class_num
        self.batch_size = batch_size
        self.mse = nn.MSELoss()
        self.cross = nn.CrossEntropyLoss()
        self.lamda = lamda
        self.classification_loss = classification_loss
        
        if classification_loss:
            self.F = torch.empty([self.batch_size*2, self.K])
        else:
            self.F = torch.empty([self.batch_size, self.K])
        nn.init.orthogonal_(self.F)
        self.encoder = nn.Sequential(nn.Linear(self.input_size,100),
                                     nn.BatchNorm1d(100),
                                     nn.ReLU(),
                                     nn.Linear(100, encoder_dim)
                                    )
        self.decoder = nn.Sequential(nn.Linear(encoder_dim, 100),
                                     nn.BatchNorm1d(100),
                                     nn.ReLU(), nn.Linear(100, self.output_size))
        
        self.head = nn.Sequential(nn.Linear(encoder_dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128,2))


    def forward(self, x):
        '''
        input: (batch_size, num_days, seq_len, feature_size)
        '''
        output = self.encoder(x)
        output = self.decoder(output)
        return output
    
    def update_kmeans_f(self, train_h):
        with torch.no_grad():
            U, sigma, VT = torch.svd(train_h)

            self.F = VT[:self.class_num].T
    
    def features(self, x):
        '''save the extracted features to form a new dataset'''
        self.eval()
        x = torch.Tensor(x)
        x = x.to(device)
        return self.encoder(x).data.cpu().numpy()

    def run_epoch(self, train_data, epoch):
        losses = []
        self.train()
        train_iterator = batch_iter(train_data, self.batch_size)
        for i, x in enumerate(train_iterator):
            self.optimizer.zero_grad()
            output = self.encoder(x)
            h = output.T
            y_pred = self.decoder(output)
            loss_reconstruct = self.mse(y_pred, x)
            loss_kmeans = torch.trace(torch.matmul(h.T, h)) - torch.trace(torch.matmul(self.F.T, torch.matmul(h.T, torch.matmul(h, self.F))))
            loss = loss_reconstruct + self.lamda / 2 * loss_kmeans
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
        if epoch and epoch % 10 == 0:
            self.update_kmeans_f(h)

        avg_train_loss = np.mean(losses)

        # Evalute Accuracy on validation set
        return avg_train_loss

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op


    def reduce_lr(self, ratio=.5):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * ratio


class _ScaledDotProductAttention(Module):
    def __init__(self, d_k: int): self.d_k = d_k

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        # scores : [bs x n_heads x q_len x q_len]
        scores = torch.matmul(q, k)

        # Scale
        scores = scores / (self.d_k ** 0.5)

        # Mask (optional)
        if mask is not None:
            scores.masked_fill_(mask, -1e9)

        # SoftMax
        # attn   : [bs x n_heads x q_len x q_len]
        attn = F.softmax(scores, dim=-1)

        # MatMul (attn, v)
        # context: [bs x n_heads x q_len x d_v]
        context = torch.matmul(attn, v)

        return context, attn

# Internal Cell


class _MultiHeadAttention(Module):
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int):
        r"""
        Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]
        """
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None):

        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        # q_s    : [bs x n_heads x q_len x d_k]
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        k_s = self.W_K(K).view(bs, -1, self.n_heads,
                               self.d_k).permute(0, 2, 3, 1)
        # v_s    : [bs x n_heads x q_len x d_v]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Scaled Dot-Product Attention (multiple heads)
        # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]
        context, attn = _ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)

        # Concat
        context = context.transpose(1, 2).contiguous().view(
            bs, -1, self.n_heads * self.d_v)  # context: [bs x q_len x n_heads * d_v]

        # Linear
        # context: [bs x q_len x d_model]
        output = self.W_O(context)

        return output, attn

# Internal Cell


class _TSTEncoderLayer(Module):
    def __init__(self, q_len: int, d_model: int, n_heads: int, d_k: Optional[int] = None, d_v: Optional[int] = None, d_ff: int = 256, res_dropout: float = 0.1,
                 activation: str = "gelu"):

        assert d_model // n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)

        # Multi-Head attention
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.batchnorm_attn = nn.BatchNorm1d(q_len)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), self._get_activation_fn(
            activation), nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.batchnorm_ffn = nn.BatchNorm1d(q_len)
    
    

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        ## Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
        src = self.batchnorm_attn(src)      # Norm: batchnorm

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)
        src = self.batchnorm_ffn(src)  # Norm: batchnorm

        return src

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        else:
            return activation()
#         raise ValueError(f'{activation} is not available. You can use "relu" or "gelu"')

# Internal Cell


class _TSTEncoder(Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, res_dropout=0.1, activation='gelu', n_layers=1):

        self.layers = nn.ModuleList([_TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                                      activation=activation) for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


# Cell
class TST(Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int, max_seq_len: Optional[int] = None, out_feature_dim: int = 50, batch_size = 50,
                 class_num = 4, lamda=1,
                 n_layers: int = 3, d_model: int = 128, n_heads: int = 16, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 d_ff: int = 256, res_dropout: float = 0.1, act: str = "gelu", fc_dropout: float = 0.,
                 y_range: Optional[tuple] = None, verbose: bool = False, **kwargs):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            res_dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            num_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.
        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.c_out, self.seq_len = c_out, seq_len

        self.class_num = class_num
        self.K = class_num
        self.batch_size = batch_size
        self.F = torch.empty([self.batch_size, self.K], device=device)
        nn.init.orthogonal_(self.F)
        self.mse = nn.MSELoss()
        self.lamda = lamda

        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len is not None and seq_len > max_seq_len:  # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(Pad1d(padding), Conv1d(
                c_in, d_model, kernel_size=tr_factor, stride=tr_factor))
            pv(f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n', verbose)
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs)  # Eq 2
            pv(f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n', verbose)
        else:
            # Eq 1: projection of feature vectors onto a d-dim vector space
            self.W_P = nn.Linear(c_in, d_model)

        # Positional encoding
        W_pos = torch.zeros((q_len, d_model), device=default_device())
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.res_dropout = nn.Dropout(res_dropout)

        # Encoder
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v,
                                   d_ff=d_ff, res_dropout=res_dropout, activation=act, n_layers=n_layers)
        self.flatten = Flatten()
        self.head_nf = q_len * d_model
        self.final_encoder = self.create_head(
            self.head_nf, out_feature_dim, fc_dropout=fc_dropout, y_range=y_range)
        # Head
        
        self.head = self.create_head(
            out_feature_dim, seq_len, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, fc_dropout=0., y_range=None, **kwargs):
        layers = [nn.Dropout(fc_dropout)] if fc_dropout else []
        layers += [nn.Linear(nf, c_out)]
        if y_range:
            layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    # x: [bs x nvars x q_len]
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:

        # Input encoding
        if self.new_q_len:
            # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
            u = self.W_P(x).transpose(2, 1)
        else:
            # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]
            u = self.W_P(x.transpose(2, 1))

        # Positional encoding
        u = self.res_dropout(u + self.W_pos)

        # Encoder
        # z: [bs x q_len x d_model]
        z = self.encoder(u)
        if self.flatten is not None:
            z = self.flatten(z)                # z: [bs x q_len * d_model]
        else:
            # z: [bs x d_model x q_len]
            z = z.transpose(2, 1).contiguous()
        z = self.final_encoder(z)
        # Classification/ Regression head
        # output: [bs x c_out]
        return self.head(z)
    
    def update_kmeans_f(self, train_h):
        with torch.no_grad():
            U, sigma, VT = torch.svd(train_h)

            self.F = VT[:self.class_num].T
    
    def run_epoch(self, train_data, epoch):
        losses = []
        self.train()
        train_iterator = batch_iter(train_data, self.batch_size)
        for i, x in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if self.new_q_len:
                # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
                u = self.W_P(x).transpose(2, 1)
            else:
                # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]
                u = self.W_P(x.transpose(2, 1))

            # Positional encoding
            u = self.res_dropout(u + self.W_pos)

            # Encoder
            # z: [bs x q_len x d_model]
            z = self.encoder(u)
            if self.flatten is not None:
                z = self.flatten(z)                # z: [bs x q_len * d_model]
            else:
                # z: [bs x d_model x q_len]
                z = z.transpose(2, 1).contiguous()
            new_sequence = self.final_encoder(z)   # output: [bs x seq_len]
            h = new_sequence.T
            y_pred = self.head(new_sequence)  # y_pred: [bs x out_feature_dim]
            raw_sequnce = x.squeeze()

            #### reconstruction loss
            loss_reconstruct = self.mse(y_pred, raw_sequnce)

            #### clsustering loss
            loss_kmeans = torch.trace(torch.matmul(h.T, h)) - torch.trace(
                torch.matmul(self.F.T, torch.matmul(h.T, torch.matmul(h, self.F))))
            
            #### shape loss
            std_raw, mean_raw = torch.std_mean(
                raw_sequnce, unbiased=False, dim=1, keepdim=True)
            std_new, mean_new = torch.std_mean(
                new_sequence, unbiased=False, dim=1, keepdim=True)

            zscore_raw = (raw_sequnce-mean_raw) / std_raw
            zscore_new = (new_sequence-mean_new) / std_new

            skew_raw = torch.mean(zscore_raw**3, dim=1, keepdim=True)
            skew_new = torch.mean(zscore_new**3, dim=1, keepdim=True)

            kurt_raw = torch.mean(zscore_raw**4, dim=1, keepdim=True) - 3.0
            kurt_new = torch.mean(zscore_new**4, dim=1, keepdim=True) - 3.0

            # std_raw, mean_raw = std_raw.ravel(), mean_raw.ravel()
            # std_new, mean_new = std_new.ravel(), mean_new.ravel()

            raw_shape_indexes = torch.cat(
                [std_raw, mean_raw, skew_raw, kurt_raw], dim=1)
            new_shape_indexes = torch.cat(
                [std_new, mean_new, skew_new, kurt_new], dim=1)

            shape_indexes_loss = self.mse(new_shape_indexes, raw_shape_indexes)

            #### total loss
            loss = loss_reconstruct + self.lamda / 2 * loss_kmeans + shape_indexes_loss
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
        if epoch and epoch % 10 == 0:
            self.update_kmeans_f(h)

        avg_train_loss = np.mean(losses)

        # Evalute Accuracy on validation set
        return avg_train_loss
    
    def features(self, x):
            # Input encoding
        self.eval()
        x = torch.Tensor(x)
        x = x.to(device)
        if self.new_q_len:
            # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
            u = self.W_P(x).transpose(2, 1)
        else:
            # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]
            u = self.W_P(x.transpose(2, 1))

        # Positional encoding
        u = self.res_dropout(u + self.W_pos)
        z = self.encoder(u)
        if self.flatten is not None:
            z = self.flatten(z)                # z: [bs x q_len * d_model]
        else:
            # z: [bs x d_model x q_len]
            z = z.transpose(2, 1).contiguous()
        z = self.final_encoder(z)

        # Encoder
        # z: [bs x q_len x d_model]
        return z.data.cpu().numpy()
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_loss_op(self, loss_op):
        self.loss_op = loss_op

    def reduce_lr(self, ratio=.5):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * ratio

# if __name__ == "__main__":

#     year = 2011
#     stocklist, datas, dates = load_data(year)
#     train_x = pre_process_data(datas, compression='downsample')
#     train_x = train_x[:,np.newaxis,:]
#     data_shape = train_x.shape

#     model = DCTR(data_shape[1])
#     optimizer = optim.AdamW(model.parameters(), lr=0.0001)
#     model.add_optimizer(optimizer)
    
#     epoch = 300
#     train_losses = []
#     savestep = 5
#     best_result = float('inf')
#     best_parameter = None
#     general_s = time.perf_counter()
#     for i in range(1,epoch+1):
#         print("Epoch: {}".format(i))
#         start_t = time.perf_counter()
#         train_loss= model.run_epoch(train_x, i)
        # print("\tAverage training loss: {:.5f}".format(train_loss))
#         end_t = time.perf_counter()
#         train_losses.append(train_loss)
#         if train_loss < best_result:
#             print('\tNew record, model saved!')
#             best_result = train_loss
#             best_parameter = model.state_dict()
#             torch.save(model.state_dict(), './experiment/dtcr.pkl')
#         print('run time: {:.3f}s'.format(end_t-start_t))
#     general_e = time.perf_counter()
#     print('run time in all: {:.3f}s'.format(general_e-general_s))
#     model.load_state_dict(best_parameter)

#     ### test
#     transformed = model.features(train_x)
#     print(transformed.shape)

    # print(device)
    # model = TST(1, data_shape[-1], data_shape[-1], 500)
    # optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    # model.add_optimizer(optimizer)
    # model.to(device)
    # model.run_epoch(train_x, 1)

