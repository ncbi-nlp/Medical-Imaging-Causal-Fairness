# due to sensitivity issues of medical data, models were trained on different servers depending on which institute has access to which data. Therefore, these evaluations
# were initially 3 separate python files. We put them together here for better sharing purpose.

import pandas as pd
import pickle
import re
from pytorch_lightning.lite import LightningLite
import skimage, torchvision
import numpy as np
import sys
import warnings
import os
import math
import random
from PIL import Image
import argparse
import time
import cv2
import copy
from tqdm import tqdm
# import matplotlib.pyplot as plt
from pprint import pprint
from torchvision import transforms
from PIL import Image
from sklearn import preprocessing, utils, metrics, decomposition
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch import autograd
from torch.autograd import Variable
from torch.utils import data
import torchvision
from torchvision.utils import save_image
from torchvision import datasets, models, transforms, utils
from RES_VAE import *
from models import *
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt


image_size = 224
batch_size = 3
epoch_num = 3
root = '../chexpertchestxrays-u20210408/'
competition_tasks = torch.ByteTensor([0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]) # one extra for no finding in competition tasks
chexpertRepetition7 = [30 if i < 15 else [9,9,9,9,9,9,8][i-15] for i in range(0, (15 + 7))]

# chexpert

def five_race():
    demo = prepare_cxr_dataset(root, 'CHEXPERT DEMO.xlsx', pd.read_csv(os.path.join(root, './CheXpert-v1.0/valid.csv')))
    race = demo['PRIMARY_RACE']
    race = np.array(race)
    ii = [np.where(race == val)[0] for val in range(0, 6)]
    ii[4] = ii[5] + ii[4]
    ii = ii[:5]
    return ii
def age_20():
    demo = prepare_cxr_dataset(root, 'CHEXPERT DEMO.xlsx', pd.read_csv(os.path.join(root, './CheXpert-v1.0/valid.csv')))
    age = demo['Age']
    age = np.array(age)//0.2
    ii = [np.where(age == val)[0] for val in range(0,5)]
    return ii
def sex():
    demo = prepare_cxr_dataset(root, 'CHEXPERT DEMO.xlsx', pd.read_csv(os.path.join(root, './CheXpert-v1.0/valid.csv')))
    sex = demo['Sex']
    # race = np.array(race)//0.1
    ii = [np.where(sex == val)[0] for val in [0,1]]
    return ii
mad = lambda l: np.sum(np.abs(np.array(l)-np.mean(l)))/len(l)

class Lite(LightningLite):
    def run(self, model_type, model_index):
        print(self.device, model_type)
        chexpertRepetition7 = [32 if i < 15 else (6 if (i == (15 + 7)-1 or i == 15 )else 4) for i in range(0, (15 + 7))]
        model, vae = load_model(model_type, path = None, vae_path = None, repetition = chexpertRepetition7)
        
        path = './weights/chexpert/{}_{}/model.pt'.format(model_type, model_index)
        model= self.setup(model)
        model.load_state_dict(torch.load(path))
        competition_tasks = torch.ByteTensor([1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        df = pd.read_csv(os.path.join(root, './CheXpert-v1.0/valid.csv'))
        test_dataset = chexpert(df, root, train = False)
        test_dataset = Data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 8)
        ground_truth, predictions = eval_model(model, model_type, test_dataset, competition_tasks, verbose = 1, device = self.device)
        return ground_truth, predictions

model_performance = {}
for model_type in ['Densenet','CausalDensenet_noise', 'VAEclassifier', 'CausalVAEclassifier_noise']:

    model_aucs = []
    model_group_dependent = []
    for index in [1,2,3,4,5]:
        ground_truth, predictions = Lite(devices='auto', accelerator="auto", precision=16).run(model_type, index)
        # break
        valid_index = [i  for i in range(len(ground_truth)) if sum(ground_truth[i]) != 0]
        ground_truth = np.array(ground_truth)[valid_index]
        predictions = np.array(predictions)[valid_index]

        model_aucs.append((metrics.roc_auc_score(ground_truth.T, predictions.T, multi_class =  'ovr', average = 'macro'), metrics.roc_auc_score(ground_truth.T, predictions.T, multi_class =  'ovr', average = 'micro')))
        race_performance = []
 
        for race in [five_race(), age_20(), sex()]:

            race = [np.concatenate([index_array + n*len(predictions[0]) for n in range(13)]) for index_array in race]
            combined_predictions = np.concatenate(predictions)
            combined_ground = np.concatenate(ground_truth)

            group_dependent_acc = []
            logits = [combined_predictions[race_index] for race_index in race]
            labels = [combined_ground[race_index] for race_index in race]
            for ind_current in range(len(logits)):
                logit_current = logits[ind_current]
                label_current = labels[ind_current]
                pos_logit_current = logit_current[(label_current == 1.0).nonzero()]
                other_logits = []
                for ind, other_logit in enumerate(logits):
                    if ind != ind_current:
                        other_logits.append(other_logit[(labels[ind] == 0.0).nonzero()])
                # print(other_logits)
                other_logits = np.concatenate(other_logits)
                average_matrix = np.subtract(pos_logit_current.reshape(-1, 1).repeat(other_logits.shape[0], 1), other_logits.reshape(-1, 1).repeat(pos_logit_current.shape[0], 1).T)
                group_dependent_acc.append(np.mean(np.where(average_matrix>0.0,np.ones_like(average_matrix), np.zeros_like(average_matrix))))
            #     break
            # break
            group_dependent_acc = [100 *v for v in group_dependent_acc]
            race_performance.append(group_dependent_acc)

        model_group_dependent.append(race_performance)
    model_performance[model_type] = (model_aucs, model_group_dependent)

model_df = {}
for k,v in model_performance.items():
    model_df[k] = {}
    model_df[k]['Macro AUC'] = np.mean(np.array(v[0][0]))
    model_df[k]['Macro AUC std'] = np.std(np.array(v[0][0]))
print('race')
for k,v in model_performance.items():
    race_dependent = [np.std(group[0])*100 for group in v[1]]
    model_df[k]['Race GDV'] = np.mean(race_dependent)
    model_df[k]['Race GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)
print('Age')
for k,v in model_performance.items():
    race_dependent = [np.std(group[1])*100 for group in v[1]]
    model_df[k]['Age GDV'] = np.mean(race_dependent)
    model_df[k]['Age GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)
print('Sex')
for k,v in model_performance.items():
    race_dependent = [np.std(group[2])*100 for group in v[1]]
    model_df[k]['Sex GDV'] = np.mean(race_dependent)
    model_df[k]['Sex GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)

pd.DataFrame(model_df).T.to_csv('chexpert_performance.csv')

# MIDRC

midrc_root = '../MIDRC/data-jpg'

class Lite(LightningLite):
    def run(self, model_type, model_index):
        print(self.device)
        model, vae = load_model(model_type, path = None, vae_path = None, repetition = repetition, out_size = 1)
        path = './weights/MIDRC/{}_{}/model.pt'.format(model_type, model_index)
        model.load_state_dict(torch.load(path, map_location = self.device))
        model = self.setup(model)
        competition_tasks = torch.ByteTensor([1])
        file_mapping = pd.read_csv('../MIDRC/filtered_final2.csv')[['Path', 'filename']]
        test_dataset = midrc_csv(pd.read_csv('../MIDRC/split/test_{}.csv'.format(model_index)))
        test_dataset = pd.merge(test_dataset,file_mapping[['Path', 'filename']], on = 'Path')
        test_dataset = midrc(test_dataset, root = midrc_root, use_demo = False, repetition = repetition )
        test_dataset = Data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers = 8)
        ground_truth, predictions = eval_model(model, model_type, test_dataset, competition_tasks, verbose = 1, device = self.device)
        return ground_truth, predictions

model_performance = {}
repetition = [276 + 180, 8, 8,8,8,8,8,8]
for model_type in ["Densenet","CausalDensenet_noise", "VAEclassifier", "CausalVAEclassifier_noise"]:
    model_aucs = []
    model_group_dependent = []
    for model_index in [1,2,3,4,5]:
        test_path = '../MIDRC/split/test_{}.csv'.format(model_index)
        r_ii, a_ii, s_ii = midrc_race(test_path), midrc_age_20(test_path), midrc_sex(test_path), 

        ground_truth, predictions = Lite(devices=[0], accelerator="auto", precision=16).run(model_type, model_index)
        fpr, tpr, thresholds = metrics.roc_curve(np.concatenate(ground_truth), np.concatenate(predictions), pos_label=1)
        step_auc = float(metrics.auc(fpr, tpr))
        model_aucs.append(metrics.auc(fpr, tpr))
        demo_performance = []

        for demo in [r_ii, s_ii, a_ii]:

            combined_predictions = np.concatenate(predictions)
            combined_ground = np.concatenate(ground_truth)

            group_dependent_acc = []
            logits = [combined_predictions[race_index] for race_index in demo]
            labels = [combined_ground[race_index] for race_index in demo]
            for ind_current in range(len(logits)):
                logit_current = logits[ind_current]
                label_current = labels[ind_current]
                pos_logit_current = logit_current[(label_current == 1.0).nonzero()]
                other_logits = []
                for ind, other_logit in enumerate(logits):
                    if ind != ind_current:
                        other_logits.append(other_logit[(labels[ind] == 0.0).nonzero()])

                other_logits = np.concatenate(other_logits)
                average_matrix = np.subtract(pos_logit_current.reshape(-1, 1).repeat(other_logits.shape[0], 1), other_logits.reshape(-1, 1).repeat(pos_logit_current.shape[0], 1).T)
                group_dependent_acc.append(np.mean(np.where(average_matrix>0.0,np.ones_like(average_matrix), np.zeros_like(average_matrix))))

            group_dependent_acc = [100 *v for v in group_dependent_acc]
            demo_performance.append(group_dependent_acc)
        model_group_dependent.append(demo_performance)
    model_performance[model_type]=(model_aucs, model_group_dependent)

model_df = {}
for k,v in model_performance.items():
    model_df[k] = {}
    model_df[k]['Macro AUC'] = np.mean(np.array(v[0]))
    model_df[k]['Macro AUC std'] = np.std(np.array(v[0]))
    print(k, np.mean(np.array(v[0])), np.std(np.array(v[0])))
print('race')
for k,v in model_performance.items():
    race_dependent = [np.std(group[0]) for group in v[1]]
    model_df[k]['Race GDV'] = np.mean(race_dependent)
    model_df[k]['Race GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)
print('sex')
for k,v in model_performance.items():
    race_dependent = [np.std(group[1]) for group in v[1]]
    model_df[k]['Sex GDV'] = np.mean(race_dependent)
    model_df[k]['Sex GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)
print('age')
for k,v in model_performance.items():
    race_dependent = [np.std(group[2]) for group in v[1]]
    model_df[k]['Age GDV'] = np.mean(race_dependent)
    model_df[k]['Age GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)

pd.DataFrame(model_df).T.to_csv('MIDRC_performance.csv')
pd.DataFrame(model_df).T


# mimic

mimic_path = '../mimic/'
repetition = [30 if i < 15 else [9,9,9,9,9,9,8][i-15] for i in range(0, (15 + 7))]

class Lite(LightningLite):
    def run(self, model_type, model_index):
        print(self.device)
        chexpertRepetition7 = [32 if i < 15 else (6 if (i == (15 + 7)-1 or i == 15 )else 4) for i in range(0, (15 + 7))]
        model, vae = load_model(model_type, path = None, vae_path = None, repetition = repetition)
        path = './weights/mimic/{}_{}/model.pt'.format(model_type, model_index)
        model.load_state_dict(torch.load(path, map_location = self.device))
        model = self.setup(model)
        competition_tasks = torch.ByteTensor([1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        test_dataset = mimic_csv(pd.read_csv('../mimic/test.csv'))
        test_dataset = mimic(test_dataset, root = '../mimic/2.0.0/', use_demo = False, mimicRepetition = repetition )
        test_dataset = Data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 8)
        ground_truth, predictions = eval_model(model, model_type, test_dataset, competition_tasks, verbose = 1, device = self.device)
        return ground_truth, predictions


model_performance = {}
for model_type in ["Densenet", "CausalDensenet_noise", "VAEclassifier", "CausalVAEclassifier_noise"]:

    model_aucs = []
    model_group_dependent = []
    for index in [1,2,3,4,5]:
        ground_truth, predictions = Lite(devices=[0], accelerator="auto", precision=16).run(model_type, index)
        ground_truth = np.array(ground_truth)[valid_index]
        predictions = np.array(predictions)[valid_index]

        model_aucs.append((metrics.roc_auc_score(ground_truth.T, predictions.T, multi_class =  'ovr', average = 'macro'), metrics.roc_auc_score(ground_truth.T, predictions.T, multi_class =  'ovr', average = 'micro')))
        demo_performance = []

        for demo in [mimic_race(), mimic_sex(), mimic_age_20()]:

            demo = [np.concatenate([index_array + n*len(predictions[0]) for n in range(14)]) for index_array in demo]
            combined_predictions = np.concatenate(predictions)
            combined_ground = np.concatenate(ground_truth)

            group_dependent_acc = []
            logits = [combined_predictions[race_index] for race_index in demo]
            labels = [combined_ground[race_index] for race_index in demo]
            for ind_current in range(len(logits)):
                logit_current = logits[ind_current]
                label_current = labels[ind_current]
                pos_logit_current = logit_current[(label_current == 1.0).nonzero()]
                other_logits = []
                for ind, other_logit in enumerate(logits):
                    if ind != ind_current:
                        other_logits.append(other_logit[(labels[ind] == 0.0).nonzero()])

                other_logits = np.concatenate(other_logits)
                average_matrix = np.subtract(pos_logit_current.reshape(-1, 1).repeat(other_logits.shape[0], 1), other_logits.reshape(-1, 1).repeat(pos_logit_current.shape[0], 1).T)
                group_dependent_acc.append(np.mean(np.where(average_matrix>0.0,np.ones_like(average_matrix), np.zeros_like(average_matrix))))

            group_dependent_acc = [100 * v for v in group_dependent_acc]
            demo_performance.append(group_dependent_acc)
        model_group_dependent.append(demo_performance)
    model_performance[model_type]=(model_aucs, model_group_dependent)


model_df = {}
for k,v in model_performance.items():
    model_df[k] = {}
    model_df[k]['Macro AUC'] = np.mean(np.array(v[0][0]))
    model_df[k]['Macro AUC std'] = np.std(np.array(v[0][0]))
print('race')
for k,v in model_performance.items():
    race_dependent = [np.std(group[0])*100 for group in v[1]]
    model_df[k]['Race GDV'] = np.mean(race_dependent)
    model_df[k]['Race GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)
print('Age')
for k,v in model_performance.items():
    race_dependent = [np.std(group[1])*100 for group in v[1]]
    model_df[k]['Age GDV'] = np.mean(race_dependent)
    model_df[k]['Age GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)
print('Sex')
for k,v in model_performance.items():
    race_dependent = [np.std(group[2])*100 for group in v[1]]
    model_df[k]['Sex GDV'] = np.mean(race_dependent)
    model_df[k]['Sex GDV std'] = np.std(race_dependent)
    print(k, np.mean(race_dependent), np.std(race_dependent), race_dependent)

pd.DataFrame(model_df).T.to_csv('mimic_performance.csv')