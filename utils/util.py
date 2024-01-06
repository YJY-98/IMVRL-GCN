#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:30:25 2023

@author: fht
"""

import copy
from numpy import linalg
import numpy as np

import optuna
import os

import pickle

import random

import sys

import time

import torch
from torch import nn
from sklearn.metrics import matthews_corrcoef

def calculate_mcc(y_true, y_pred_prob, thresholds):
    mcc_values = []

    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        mcc_values.append(mcc)

    return mcc_values

def setSeed(seed):
    os.environ['PYTHONASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def makeDir(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    return


def getactivation(act):
    if False:
        sys.exit('Wrong!')
    elif act == 'ReLU':
        activation = nn.ReLU
    elif act == 'LeakyReLU':
        activation = nn.LeakyReLU
    else:
        sys.exit(f'wrong act: {act}')
    return activation


def getOpt():
    pathName = '/mnt/fht/fuhaitao/cancerGenePrediction/Program/Output/CPDB_datasets/optuna/'
    fileName_ls = ['ChebConv_residual_0715_0935_study.db','ChebConv_cat_0714_2124_study.db',
                'ChebConv_average_0715_0139_128_study.db','Transformer_0714_1704_study.db',
                'ChebConv_last_0714_1635_study.db','ChebConv_average_0717_2229_256_study.db']
    df_dict=dict()
    for fileName in fileName_ls:
        storage = os.path.join(pathName, fileName)
        study = optuna.create_study(storage=f'sqlite:///{storage}', 
                                    study_name='study', 
                                    direction='maximize', 
                                    load_if_exists=True)
    
        df = study.trials_dataframe()
        # df.to_csv(storage.rstrip('db')+'csv', float_format='%.6f')
    
        best_trial = study.best_trial
        print(f'{fileName}: \nBest trial: \nNumber: {best_trial.number}; Value: {best_trial.value}\n\n\n')
        df_dict[fileName] = df
        df_dict[f'{fileName}_value'] = best_trial.value
    for key, value in best_trial.params.items():
        print(f'{key}: {value}')
    return df
# End