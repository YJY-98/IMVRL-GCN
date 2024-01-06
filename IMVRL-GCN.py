#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% Import
import copy
import math
import numpy as np
import optuna
import pandas as pd
import pickle
import random
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric
from torch_geometric.nn import ChebConv

# from torch_geometric.utils import dropout_adj, remove_self_loops, add_self_loops
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import add_self_loops

import argparse
# from HyperParameters import setactivation
from utils.util import getactivation
from utils.util import setSeed



warnings.filterwarnings('ignore')

# %% Args
def Args():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument('--dev', default='cuda:0',
                        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--epochs', type=int, default=500,
                        choices=[1, 2, 10, 20, 50, 100, 500, 1000, 1500, 2500, 3000])
    parser_args = parser.parse_args()
    return parser_args


# %% load data
def load_datasets(args):
    device = args.dev
    
    data = torch.load(f'./data/CPDB_datasets.pkl')
    data = data.to(device)
    Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(
        torch.FloatTensor).to(device)
    y_all = np.logical_or(data.y, data.y_te)
    mask_all = np.logical_or(data.mask, data.mask_te)

    with open(f'./data/k_sets.pkl', 'rb') as handle:
        k_sets = pickle.load(handle)

    pb, _ = remove_self_loops(data.edge_index)
    pb, _ = add_self_loops(pb)
    data.edge_index = copy.deepcopy(pb)
    E = data.edge_index
    
    return data, Y, mask_all, k_sets, pb, E


# %% Model


class GNNEncoder(nn.Module):
    def __init__(self, nFeat, nHid, args, outAct=lambda x: x):
        super(GNNEncoder, self).__init__()
        # 这里，层数可以作为一个参数，待修改

        self.Dropout = nn.Dropout(0.5)

        self.convLs = nn.ModuleList()

        conv = nn.ModuleList([ChebConv(nFeat, nHid, K=2)])
        conv.append(nn.ReLU())
        conv.append(nn.Dropout(0.5))
        self.convLs.append(conv)


        conv = nn.ModuleList([ChebConv(nHid, nHid, K=2)])
        conv.append(nn.ReLU())
        conv.append(nn.Dropout(0.5))
        self.convLs.append(conv)
        
        self.outAct = outAct
        return

    def forward(self, x, edge_index):

        x = self.Dropout(x)

        xLs = []
        for iLayer in range(len(self.convLs)):
            edge_temp, _ = dropout_adj(edge_index, p=0.8,
                                       force_undirected=True,
                                       num_nodes=x.size()[0],
                                       training=self.training)
            x = self.convLs[iLayer][0](x, edge_temp)
            for subLayer in self.convLs[iLayer][1:]:
                x = subLayer(x)
                
            xLs.append(x)

        xOut = sum(xLs) / len(xLs)
        xOut = self.outAct(xOut)

        return xOut


class Generator(nn.Module):
    def __init__(self, args, features):
        super(Generator, self).__init__()
        n_modality = args.n_modality
        nFeat = [temp.shape[1] for temp in features]
        self.args = args
        self.encoder = nn.ModuleList(
        [GNNEncoder(nFeat[i], 256, args) for i in range(n_modality)])
        return

    def forward(self, features, edge_index):
        args = self.args
        out = [self.encoder[i](features[i], edge_index)
               for i in range(len(features))]
        return out


class Discriminator(nn.Module):
    def __init__(self, args, outAct=lambda x: x):
        super(Discriminator, self).__init__()
        self.n_modality = n_modality = args.n_modality
        self.interactionWeight = torch.nn.Parameter(
            torch.randn(n_modality**2, 256, 256))
        self.outAct = outAct
        return

    def forward(self, x):
        output = []
        for i in range(len(x)):
            for j in range(len(x)):
                ind = i * self.n_modality + j
                out = torch.matmul(x[i], self.interactionWeight[ind])
                out = torch.mul(out, x[j]).sum(1)
                out = self.outAct(out)
                output.append(out)
        return output


class MultiLayerPerceptron(nn.Module):
    def __init__(self, args, nFeat, nHid, nOut, outAct=lambda x: x):
        super(MultiLayerPerceptron, self).__init__()

        self.MLP = nn.ModuleList()
        self.MLP.append(nn.Linear(nFeat, nOut))
        self.outAct = outAct
        return

    def forward(self, x, edge_index):
        for layer in self.MLP:
            x = layer(x)
        out = self.outAct(x)
        return out


class Predictor(nn.Module):
    def __init__(self, args, nFeat, nHid, edge_index, outAct=lambda x: x):
        super(Predictor, self).__init__()

        self.args = args   
        pb, _ = remove_self_loops(edge_index)
        self.pb, _ = add_self_loops(pb)

        self.encoder = GNNEncoder(nFeat, nHid, args)
        self.MLPs = MultiLayerPerceptron(args, nHid, int(nHid/2), 1)
        
    def forward(self, x, edge_index):

        out = self.encoder(x, edge_index)
        out = self.MLPs(out, edge_index)
        
        return out


# %% Train

class Train(object):
    def __init__(self, args):
        self.args = args
        return
    
    def trainModel(self, data, Y, tr_mask, te_mask):
        x = data.x
        features = [x[:, i:i+16] for i in range(0,x.shape[1],16)]
        edge_index = data.edge_index
        
        args = self.args
        
        device = args.dev
        epochs = args.epochs
        
        args.n_modality = n_modality = len(features)
        # 初始化共享生成器
        shared_G = Generator(args, features)
        shared_G = shared_G.to(device)
        optimizer_shared_G = torch.optim.Adam(
            shared_G.parameters(), lr=0.0005, betas=(0.5, 0.999),
            weight_decay=5e-5)

        # 初始化特异性生成器
        specif_G = Generator(args, features)
        specif_G = specif_G.to(device)
        optimizer_specif_G = torch.optim.Adam(
            specif_G.parameters(), lr=0.0005, betas=(0.5, 0.999),
            weight_decay=5e-5)

        # 初始化鉴别器
        discri_M = Discriminator(args)
        discri_M = discri_M.to(device)
        optimizer_Discrimi = torch.optim.Adam(
            discri_M.parameters(), lr=0.005, betas=(0.5, 0.999),
            weight_decay=5e-5)

        ganLossFn = nn.BCEWithLogitsLoss()

        nFeat = 256 + len(features) * 256
        nHid = 256
        predModel = Predictor(args, nFeat, nHid, edge_index)
        predModel = predModel.to(device)
        optimizer_predict = torch.optim.Adam(
            predModel.parameters(), lr=0.005, betas=(0.5, 0.999),
            weight_decay=5e-5)
        predLossFn = nn.BCEWithLogitsLoss()

        epoch_ls = []
        AUC__tr_ls = []
        AUPR_tr_ls = []

        AUC__te_ls = []
        AUPR_te_ls = []

        try:
            epoch = 1
            for epoch in range(1, epochs+1):
                shared_G.train()
                specif_G.train()
                discri_M.eval()
                predModel.train()

                # 生成步，优化生成器
                shared_out = shared_G(features, edge_index)
                shared_out_tr = [temp[tr_mask] for temp in shared_out]
                sharedDisOut_G = torch.cat(discri_M(shared_out_tr), dim=0)

                specif_out = specif_G(features, edge_index)
                specif_out_tr = [temp[tr_mask] for temp in specif_out]
                specifDisOut_G = torch.cat(discri_M(specif_out_tr), dim=0)

                # Loss measures generator's ability to fool the discriminator
                valid = torch.ones_like(sharedDisOut_G)
                fake = torch.zeros_like(sharedDisOut_G)

                sharedLoss_G = ganLossFn(sharedDisOut_G, valid)
                specifLoss_G = ganLossFn(specifDisOut_G, fake)
                loss_G = (sharedLoss_G + specifLoss_G) / 2.0

                # calculate predictions
                shared_ave = sum(shared_out) / len(shared_out)
                featuresC = [shared_ave] + specif_out
                generate_emb = torch.cat(featuresC, dim=1)
                predictions = predModel(generate_emb, edge_index)
                predLoss = predLossFn(predictions[tr_mask], Y[tr_mask])

                # optimizer generator and predModel
                optimizer_shared_G.zero_grad()
                optimizer_specif_G.zero_grad()
                optimizer_predict.zero_grad()

                totalLoss_G = 0.3 * loss_G + 0.7 * predLoss
                totalLoss_G.backward()

                optimizer_shared_G.step()
                optimizer_specif_G.step()
                optimizer_predict.step()
                
                shared_G.eval()
                specif_G.eval()
                discri_M.train()
                predModel.train()
                
                # 判别步，优化判别器
                shared_out = shared_G(features, edge_index)
                sharedDisOut_D = torch.cat(
                    discri_M([temp.detach() for temp in shared_out_tr]), dim=0)

                specif_out = specif_G(features, edge_index)
                specifDisOut_D = torch.cat(
                    discri_M([temp.detach() for temp in specif_out_tr]), dim=0)

                sharedLoss_D = ganLossFn(sharedDisOut_D, fake)
                specifLoss_D = ganLossFn(specifDisOut_D, valid)
                loss_D = (sharedLoss_D + specifLoss_D) / 2.0

                shared_ave = sum(shared_out) / len(shared_out)
                featuresC = [shared_ave] + specif_out
                generate_emb = torch.cat(featuresC, dim=1)
                predictions = predModel(generate_emb, edge_index)
                
                predLoss = predLossFn(predictions[tr_mask], Y[tr_mask])

                optimizer_Discrimi.zero_grad()
                optimizer_predict.zero_grad()

                totalLoss_D = 0.3 * loss_D + 0.7 * predLoss
                totalLoss_D.backward()

                optimizer_Discrimi.step()
                optimizer_predict.step()
                
                if (True in np.isnan(predictions.detach().cpu().numpy().flatten())):
                    temp_x = 0
                    sys.exit(f'epoch: {epoch}, nan is in predictions')
                AUC__tr = roc_auc_score(Y[tr_mask].cpu().numpy().flatten(),
                                        predictions[tr_mask].detach().cpu().numpy().flatten())
                AUPR_tr = average_precision_score(Y[tr_mask].cpu().numpy().flatten(),
                                                  predictions[tr_mask].detach().cpu().numpy().flatten())
                
                shared_G.eval()
                specif_G.eval()
                discri_M.eval()
                predModel.eval()
                with torch.no_grad():
                    shared_out = shared_G(features, edge_index)
                    specif_out = specif_G(features, edge_index)

                    shared_ave = sum(shared_out) / len(shared_out)
                    featuresC = [shared_ave] + specif_out
                    generate_emb = torch.cat(featuresC, dim=1)

                    predictions = predModel(generate_emb, edge_index)
                    
                if (True in np.isnan(predictions.cpu().numpy().flatten())):
                    temp_x = 0
                    sys.exit(f'epoch: {epoch}, nan is in predictions')
                AUC__te = roc_auc_score(Y[te_mask].cpu().numpy().flatten(), 
                                        predictions[te_mask].detach().cpu().numpy().flatten())
                AUPR_te = average_precision_score(Y[te_mask].cpu().numpy().flatten(), 
                                                  predictions[te_mask].detach().cpu().numpy().flatten())       

                if (epoch == 1) or (epoch % 10 == 0):
                    epoch_ls.append(epoch)
                    
                    AUC__tr_ls.append(AUC__tr)
                    AUPR_tr_ls.append(AUPR_tr)

                    AUC__te_ls.append(AUC__te)
                    AUPR_te_ls.append(AUPR_te)
                    
                    print(f'epoch: {epoch}')
                    print(f'-train-\nAUC: {AUC__tr:.4f}, AUPR: {AUPR_tr:.4f}')
                    print(f'-test-\nAUC: {AUC__te:.4f}, AUPR: {AUPR_te:.4f}')
        except KeyboardInterrupt:
            pass
        
        self.epoch_ls = epoch_ls
        self.AUC__tr_ls = AUC__tr_ls
        self.AUPR_tr_ls = AUPR_tr_ls

        self.AUC__te_ls = AUC__te_ls
        self.AUPR_te_ls = AUPR_te_ls

    
        return
        

# %% Experiment

class Experiment(object):
    def __init__(self, args, data, Y, mask_all, k_sets):
        self.args = args
        self.runFold(data, Y, k_sets)
        return
              
    def subParaFun(self, data, Y, k_sets):
        args = self.args
        
        epochs = args.epochs

        numResult = math.floor(float(epochs)/10) + 1

        AUC__tr_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_tr_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_te_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_final = np.zeros((numResult, 1+10+2))
        AUPR_te_final = np.zeros((numResult, 1+10+2))
        
        columns_Run = ['epoch'] + [f'Run{iRun+1}'
                                   for iRun in range(10)] + ['mean', 'std']
        columns_fold = ['epoch'] + [f'fold{fold+1}'
                                   for fold in range(5)] + ['mean', 'std']
        
        iRun = 0
        for iRun in range(10):
            timeRun = time.time()
            kFold = 0
            for kFold in range(5):
                timeFold = time.time()
                print(f'Run: {iRun+1} \nFold: {kFold+1}')
                _, _, tr_mask, te_mask = k_sets[iRun][kFold]
                trainObj = Train(args)
                trainObj.trainModel(data, Y, tr_mask, te_mask)
                
                AUC__tr_ls = trainObj.AUC__tr_ls
                AUPR_tr_ls = trainObj.AUPR_tr_ls
                AUC__te_ls = trainObj.AUC__te_ls
                AUPR_te_ls = trainObj.AUPR_te_ls
                
                AUC__tr_Fold[iRun,:,kFold+1] = np.array(AUC__tr_ls)
                AUPR_tr_Fold[iRun,:,kFold+1] = np.array(AUPR_tr_ls)
                AUC__te_Fold[iRun,:,kFold+1] = np.array(AUC__te_ls)
                AUPR_te_Fold[iRun,:,kFold+1] = np.array(AUPR_te_ls)
                
                elapsedTime = round((time.time() - timeFold) / 60, 3)
                print(f'Fold time: {elapsedTime} minutes')
                
                break

            AUC__tr_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUC__tr_Fold[iRun,:,-2] = np.mean(AUC__tr_Fold[iRun,:,1:5+1], axis=1)
            AUC__tr_Fold[iRun,:,-1] = np.std(AUC__tr_Fold[iRun,:,1:5+1], axis=1)
            AUPR_tr_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUPR_tr_Fold[iRun,:,-2] = np.mean(AUPR_tr_Fold[iRun,:,1:5+1], axis=1)
            AUPR_tr_Fold[iRun,:,-1] = np.std(AUPR_tr_Fold[iRun,:,1:5+1], axis=1)
            
            AUC__te_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUC__te_Fold[iRun,:,-2] = np.mean(AUC__te_Fold[iRun,:,1:5+1], axis=1)
            AUC__te_Fold[iRun,:,-1] = np.std(AUC__te_Fold[iRun,:,1:5+1], axis=1)
            AUPR_te_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
            AUPR_te_Fold[iRun,:,-2] = np.mean(AUPR_te_Fold[iRun,:,1:5+1], axis=1)
            AUPR_te_Fold[iRun,:,-1] = np.std(AUPR_te_Fold[iRun,:,1:5+1], axis=1)
                
            elapsedTime = round((time.time() - timeRun) / 3600, 2)
            print(f'Run time: {elapsedTime} hours')
            break
            
        AUC__te_final[:, 0] = trainObj.epoch_ls
        AUC__te_final[:, iRun+1] = AUC__te_Fold[iRun, :, -2]
        AUC__te_final[:, -2] = np.mean(AUC__te_final[:, 1: 10+1], axis=1)
        AUC__te_final[:, -1] = np.std(AUC__te_final[:, 1: 10+1], axis=1)
        
        AUPR_te_final[:, 0] = trainObj.epoch_ls
        AUPR_te_final[:, iRun+1] = AUPR_te_Fold[iRun, :, -2]
        AUPR_te_final[:, -2] = np.mean(AUPR_te_final[:, 1: 10+1], axis=1)
        AUPR_te_final[:, -1] = np.std(AUPR_te_final[:, 1: 10+1], axis=1)
            
        # self.trainObj = trainObj
        self.AUC__tr_Fold = AUC__tr_Fold
        self.AUPR_tr_Fold = AUPR_tr_Fold
        self.AUC__te_Fold = AUC__te_Fold
        self.AUPR_te_Fold = AUPR_te_Fold
        self.AUC__te_final = AUC__te_final
        self.AUPR_te_final = AUPR_te_final
        return
        
    def runFold(self, data, Y, k_sets):
        args = self.args
        
        epochs = args.epochs

        numResult = math.floor(float(epochs)/10) + 1

        AUC__tr_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_tr_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_Fold = np.zeros((10, numResult, 1+5+2))
        AUPR_te_Fold = np.zeros((10, numResult, 1+5+2))

        AUC__te_final = np.zeros((numResult, 1+10+2))
        AUPR_te_final = np.zeros((numResult, 1+10+2))
        
        columns_Run = ['epoch'] + [f'Run{iRun+1}'
                                   for iRun in range(10)] + ['mean', 'std']
        columns_fold = ['epoch'] + [f'fold{fold+1}'
                                   for fold in range(5)] + ['mean', 'std']
        
        iRun = 0
        for iRun in range(10):
            
            timeRun = time.time()
            kFold = 0
            for kFold in range(5):
                timeFold = time.time()
                print(f'Run: {iRun+1} \nFold: {kFold+1}')
                _, _, tr_mask, te_mask = k_sets[iRun][kFold]
                trainObj = Train(args)
                trainObj.trainModel(data, Y, tr_mask, te_mask)
                 
                AUC__tr_ls = trainObj.AUC__tr_ls
                AUPR_tr_ls = trainObj.AUPR_tr_ls
                AUC__te_ls = trainObj.AUC__te_ls
                AUPR_te_ls = trainObj.AUPR_te_ls
                
                AUC__tr_Fold[iRun,:,kFold+1] = np.array(AUC__tr_ls)
                AUPR_tr_Fold[iRun,:,kFold+1] = np.array(AUPR_tr_ls)
                AUC__te_Fold[iRun,:,kFold+1] = np.array(AUC__te_ls)
                AUPR_te_Fold[iRun,:,kFold+1] = np.array(AUPR_te_ls)

                AUC__tr_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
                AUC__tr_Fold[iRun,:,-2] = np.mean(AUC__tr_Fold[iRun,:,1:kFold+2], axis=1)
                AUC__tr_Fold[iRun,:,-1] = np.std(AUC__tr_Fold[iRun,:,1:kFold+2], axis=1)
                AUPR_tr_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
                AUPR_tr_Fold[iRun,:,-2] = np.mean(AUPR_tr_Fold[iRun,:,1:kFold+2], axis=1)
                AUPR_tr_Fold[iRun,:,-1] = np.std(AUPR_tr_Fold[iRun,:,1:kFold+2], axis=1)
                
                AUC__te_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
                AUC__te_Fold[iRun,:,-2] = np.mean(AUC__te_Fold[iRun,:,1:kFold+2], axis=1)
                AUC__te_Fold[iRun,:,-1] = np.std(AUC__te_Fold[iRun,:,1:kFold+2], axis=1)
                AUPR_te_Fold[iRun,:,0] = np.array(trainObj.epoch_ls)
                AUPR_te_Fold[iRun,:,-2] = np.mean(AUPR_te_Fold[iRun,:,1:kFold+2], axis=1)
                AUPR_te_Fold[iRun,:,-1] = np.std(AUPR_te_Fold[iRun,:,1:kFold+2], axis=1)
                
                
                ind_max = np.argmax(AUC__te_Fold[iRun, :, -2])
                epoch_max = AUC__te_Fold[iRun, ind_max, 0]
                AUC__tr_max = round(AUC__tr_Fold[iRun, ind_max, -2], 4)
                AUPR_tr_max = round(AUPR_tr_Fold[iRun, ind_max, -2], 4)
                
                AUC__te_max = round(AUC__te_Fold[iRun, ind_max, -2], 4)
                AUPR_te_max = round(AUPR_te_Fold[iRun, ind_max, -2], 4)
                
                outTail = f'iRun{iRun+1}_kFold{kFold+1}_epoch{epoch_max}_maxAUC{AUC__tr_max}'
                outName = f'./output/train_{outTail}.csv'
                AUC__tr_pd = pd.DataFrame(AUC__tr_Fold[iRun], columns=columns_fold)
                AUC__tr_pd.to_csv(outName, float_format='%.4f')
                
                outTail = f'iRun{iRun+1}_kFold{kFold+1}_epoch{epoch_max}_maxAUPR{AUPR_tr_max}'
                outName = f'./output/train_{outTail}.csv'
                AUPR_tr_pd = pd.DataFrame(AUPR_tr_Fold[iRun], columns=columns_fold)
                AUPR_tr_pd.to_csv(outName, float_format='%.4f')
                
                outTail = f'iRun{iRun+1}_kFold{kFold+1}_epoch{epoch_max}_maxAUC{AUC__te_max}'
                outName = f'./output/test_{outTail}.csv'
                AUC__te_pd = pd.DataFrame(AUC__te_Fold[iRun], columns=columns_fold)
                AUC__te_pd.to_csv(outName, float_format='%.4f')
                
                outTail = f'iRun{iRun+1}_kFold{kFold+1}_epoch{epoch_max}_maxAUPR{AUPR_te_max}'
                outName = f'./output/test_{outTail}.csv'
                AUPR_te_pd = pd.DataFrame(AUPR_te_Fold[iRun], columns=columns_fold)
                AUPR_te_pd.to_csv(outName, float_format='%.4f')
                
                elapsedTime = round((time.time() - timeFold) / 60, 3)
                print(f'Fold time: {elapsedTime} minutes')
                # break
            
            AUC__te_final[:, 0] = trainObj.epoch_ls
            AUC__te_final[:, iRun+1] = AUC__te_Fold[iRun, :, -2]
            AUC__te_final[:, -2] = np.mean(AUC__te_final[:, 1: iRun+2], axis=1)
            AUC__te_final[:, -1] = np.std(AUC__te_final[:, 1: iRun+2], axis=1)
            
            AUPR_te_final[:, 0] = trainObj.epoch_ls
            AUPR_te_final[:, iRun+1] = AUPR_te_Fold[iRun, :, -2]
            AUPR_te_final[:, -2] = np.mean(AUPR_te_final[:, 1: iRun+2], axis=1)
            AUPR_te_final[:, -1] = np.std(AUPR_te_final[:, 1: iRun+2], axis=1)
            
            ind_max = np.argmax(AUC__te_final[:, -2])
            epoch_max = AUC__te_final[ind_max, 0]
            AUC__final_max = round(AUC__te_final[ind_max, -2], 4)
            AUPR_final_max = round(AUPR_te_final[ind_max, -2], 4)
            
            outTail = f'iRun{iRun+1}_epoch{epoch_max}_maxAUC{AUC__final_max}'
            outName = f'./output/finalTest_{outTail}.csv'
            AUC_te_final_pd = pd.DataFrame(AUC__te_final, columns=columns_Run)
            AUC_te_final_pd.to_csv(outName, float_format='%.4f')
            
            outTail = f'iRun{iRun+1}_epoch{epoch_max}_maxAUPR{AUPR_final_max}'
            outName = f'./output/finalTest_{outTail}.csv'
            AUPR_te_final_pd = pd.DataFrame(AUPR_te_final, columns=columns_Run)
            AUPR_te_final_pd.to_csv(outName, float_format='%.4f')
            
            elapsedTime = round((time.time() - timeRun) / 3600, 2)
            print(f'Run time: {elapsedTime} hours')
            
            # break
        self.trainObj = trainObj
        return
    

# %% Main
if __name__ == '__main__':
    args = Args()

    data, Y, mask_all, k_sets, pb, E = load_datasets(args)
    expObj = Experiment(args, data, Y, mask_all, k_sets)
    
