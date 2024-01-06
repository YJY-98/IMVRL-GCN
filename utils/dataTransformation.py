#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 17:29:37 2023

@author: fht
"""

import os
import sys
import torch

import torch_geometric
from torch_geometric.data import Data
# print(os.getcwd())
# sys.exit()
def dataTrans_17(ppi):
    assert torch_geometric.__version__.startswith('1')
    prefix = f'../Dataset/{ppi}/{ppi}'
    fileName = f'{prefix}_data.pkl'
    data = torch.load(fileName)
    data_dict = data.to_dict()
    
    outName = f'{prefix}_data_dict.pt'
    if not os.path.exists(outName):
        torch.save(data_dict, outName)
    return data
def dataTrans_20(ppi):
    assert torch_geometric.__version__.startswith('2')
    prefix = f'../Dataset/{ppi}/{ppi}'
    fileName = f'{prefix}_data_dict.pt'
    data_dict = torch.load(fileName)
    data = Data.from_dict(data_dict)
    # torch.save(data, 'data.pt')
    return data

if __name__ == '__main__':
    ppi = sys.argv[1]
    try:
        data = dataTrans_17(ppi)
    except:
        data = dataTrans_20(ppi)
# %% End