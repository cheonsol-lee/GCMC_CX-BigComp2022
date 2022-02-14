import time
import datetime

# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


"""Training GCMC model on the MovieLens data set.
The script loads the full graph to the training device.
"""
import os, time
import argparse
import logging
import random
import string
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from data_rotten import RottenMovie
from utils import get_activation, get_optimizer, torch_total_param_num, torch_net_info, MetricLogger

import easydict

args = easydict.EasyDict({ 
    "data_name":                      "rotten", 
    "use_one_hot_fea":                False,
    "gpu":                            0,
    "seed":                           123,
    "data_test_ratio":                0.1,
    "data_valid_ratio":               0.1,
    "model_activation":               'leaky',
    "gcn_dropout":                    0.5,
    "gcn_agg_norm_symm":              True,
    "gcn_agg_units":                  32,
    "gcn_agg_accum":                  'sum',
    "gcn_out_units":                  32, # 64, 128
    "gen_r_num_basis_func":           2,
    "train_max_epoch":                300,
    "train_log_interval":             5,
    "train_valid_interval":           5,
    "train_optimizer":                'adam',
    "train_grad_clip":                1.0,
    "train_lr":                       0.01,
    "train_min_lr":                   0.0008,
    "train_lr_decay_factor":          0.5,
    "train_decay_patience":           25,
    "train_early_stopping_patience":  50,
    "share_param":                    False,
    "mix_cpu_gpu":                    False,
    "minibatch_size":                 40000,
    "num_workers_per_gpu":            8,
    "device":                         'cpu',
    "save_dir":                       './save/',
    "save_id":                        1,
    "train_max_iter":                 300
})


np.random.seed(args.seed)
th.manual_seed(args.seed)

if th.cuda.is_available():
    th.cuda.manual_seed_all(args.seed)

from train import train

dataset = RottenMovie(                 
             train_data='./data/trainset_filtered.csv',
             test_data='./data/testset_filtered.csv',
             movie_data = './data/movie_info.csv',
             user_data = './data/user_info.csv',
             emotion=False,
             sentiment=False,

             name='rotten', 
             device='cpu', 
             mix_cpu_gpu=False,
             use_one_hot_fea=False, 
             symm=True,
             valid_ratio=0.1,
             )


dataset_es = RottenMovie(                 
             train_data='./data/trainset_filtered.csv',
             test_data='./data/testset_filtered.csv',
             movie_data = './data/movie_info.csv',
             user_data = './data/user_info.csv',
             emotion=True,
             sentiment=True,

             name='rotten', 
             device='cpu', 
             mix_cpu_gpu=False,
             use_one_hot_fea=False, 
             symm=True,
             valid_ratio=0.1,
             )

args.rating_vals = dataset.rating_values

args.gcn_dropout = 0.50

if __name__ == '__main__':
    bests=100
    bests_es=100
    for dim in [64]:
        args.gcn_out_units = dim
        for agg in [64]:
            args.gcn_agg_units = agg
            for lr in [0.006*i for i in range(10)]:
                args.train_lr = lr
                args.save_dir = f'./test/test'
                args.save_id = 'new_feature'
                best = train(args, dataset)
                print("****************************")
                args.save_dir = f'./test/test_es'
                args.save_id = 'new_feature_es'
                best_es = train(args, dataset_es)

                print(best,'  VS  ', best_es)
                if bests>best:
                    bests = best
                if bests_es>best_es:
                    bests_es=best_es

    print(bests,'  VS  ', bests_es)