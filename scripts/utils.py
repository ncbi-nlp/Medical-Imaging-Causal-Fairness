import os
import math
import random
import re
import torch.nn as nn
from torch.utils import data
import argparse
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import preprocessing, metrics
import importlib
from RES_VAE import *
from pathlib import Path
import compare_auc_delong_xu 
from tqdm import tqdm
 
image_size = 224

def prepare_cxr_dataset(root, demo, data_df):
    f = pd.read_excel(os.path.join(root, demo), engine = 'openpyxl')
    df = data_df
    f = f.fillna(0) 
    df['PATIENT'] = df['Path'].map(lambda p: str(re.search('patient\d\d\d\d\d',p)[0]))
    df = df.merge(f, on='PATIENT')
    df['Age'] = df['Age']/100
    df['Sex'] = df['Sex'].map(lambda p: sex2num[p])
    race_num = [race2num[r][0] for r in list(df['PRIMARY_RACE'])]
    # race_num = [race_label_helper(race2num[r]) for r in list(df['PRIMARY_RACE'])]
    # race_num = le.transform(race_num)
    # df['PRIMARY_RACE1'] = df['PRIMARY_RACE'].map(lambda p: le.transform([race_label_helper(race2num[p])]))
    df['PRIMARY_RACE'] = race_num
    df['ETHNICITY'] = df['ETHNICITY'].map(lambda p: eth2num[p])
    return df

sex2num = {'Female':0, 'Male':1, 'Unknown':2}
race2num = {0: (0, 0),
 'American Indian or Alaska Native': (4,0),
 'Asian': (3,0),
 'Asian - Historical Conv': (3,0),
 'Asian, Hispanic': (3,1),
 'Asian, non-Hispanic':(3,2),
 'Black or African American': (2,0),
 'Black, Hispanic':(2,1),
 'Black, non-Hispanic':(2,2),
 'Native American, Hispanic':(4,1),
 'Native American, non-Hispanic':(4,2),
 'Native Hawaiian or Other Pacific Islander':(5,0),
 'Other':(0,0),
 'Other, Hispanic':(0,1),
 'Other, non-Hispanic':(0,2),
 'Pacific Islander, Hispanic':(5,1),
 'Pacific Islander, non-Hispanic':(5,2),
 'Patient Refused':(0,0),
 'Race and Ethnicity Unknown':(0,0),
 'Unknown':(0,0),
 'White':(1,0),
 'White or Caucasian':(1,0),
 'White, Hispanic':(1,1),
 'White, non-Hispanic':(1,2)}
eth2num={0:1,
 'Hispanic':0,
 'Hispanic/Latino':0,
 'Non-Hispanic/Non-Latino':2,
 'Not Hispanic':2,
 'Patient Refused':1,
 'Unknown':1}
view2num = {'Frontal':0, 'Lateral': 1}
race_label_helper = lambda x: x[0] + 9*x[1]
le = preprocessing.LabelEncoder()
le.fit([a + b*9 for a,b in list(race2num.values())])



def predict_batch(model, model_type, images, vae = None, label_tensor = None, inference = False, discriminator = None):
    result = {}
    if 'Densenet' in model_type:
        result = model(images, label_tensor)
    elif 'VAE' in model_type:
        result = model(images, label_tensor, discriminator)
    disease_classes = result['disease_classes']
    
    if inference:
        return disease_classes, result['z']
    return result

def load_model(model_type, repetition, path = None, vae_path = None, out_size = 15):
    model = None
    model_module = importlib.import_module("models")
    if 'Densenet' in model_type:
        model_class = getattr(model_module, 'CausalDensenet')
        model = model_class(causal = 'Causal' in model_type, repetition = repetition, out_size = out_size)
    elif model_type == 'fixed_vae_classifier':
        model = classifier_head()
    else:
        model_class = getattr(model_module, 'CausalVAEclassifier')
        causal = 'Causal' in model_type
        dis = 'Dis' in model_type
        adv = 'Adv' in model_type
        before = 'Before' in model_type
        seperate = 'Sep' in model_type
        print('Causal: ', causal, 'Disentangle: ', dis, 'Adversarial: ', adv, 'Before: ', before, 'Seperate: ', seperate)
        model = model_class(causal = causal, disentangle = dis, adversarial = adv, 
            before = before, seperate = seperate, repetition = repetition, out_size = out_size)

    if path:
        # print('loading model')
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint)
    vae = None
    if model_type == 'fixed_vae_classifier' or model_type  == 'vae':
        vae = VAE()
        if vae_path:
            print('loading vae')
            checkpoint = torch.load(vae_path, map_location='cpu')
            vae.load_state_dict(checkpoint)
    return model, vae


        
tforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}



def labels2tensor(labels, repetition):
    output = []
    for label, rep in zip(labels, repetition):
        output.append(np.array([label] * rep))
    return np.concatenate(output)


class chexpert(data.Dataset):
    def __init__(self, df, root, use_demo = False, train = True, chexpertRepetition = None):
        self.datalist = np.array(list(df.itertuples(index=False, name=None)))
        
        self.imgs = [os.path.join(root, k[0]) for k in self.datalist]
        self.labels = df.iloc[:,5:19].copy().fillna(0)
        self.labels = self.labels.replace(-1,1)
        self.labels['competition_no_findings'] = np.array((self.labels['Cardiomegaly'] + 
            self.labels['Edema'] + self.labels['Pleural Effusion'] + 
            self.labels['Atelectasis'] + self.labels['Consolidation'])<1)*1.0
        # self.labels.to_csv('test.csv')
        self.labels = torch.FloatTensor(self.labels.to_numpy())
        self.train = train
        self.repetition = chexpertRepetition
        self.use_demo = use_demo
        if use_demo:
            self.demos = (df.iloc[:,[1,2,-2,-1]] if len(self.repetition) == 15+8 else df.iloc[:,[1,2,-2]]).to_numpy()
            # print(self.demos)
        
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path).convert('RGB')/255
        img = tforms['train'](img) if self.train else tforms['val'](img) 
        # print(img)
        if self.use_demo:
            demo = self.demos[idx]
            #print(demo)
            race = [0,0,0,0,0]
            race[int(demo[2])-1] = 1
            if int(demo[2]) == 0:
                race = [1,1,1,1,1]
            demo = np.concatenate([demo[0:1], race, demo[1:2], demo[3:4]]) if len(self.repetition) == 15+8 else np.concatenate([demo[0:1], race, demo[1:2]])
            # print(len(demo))
            
            tensor = labels2tensor(list(label.numpy()) + list(demo), self.repetition)
            
            # print(type(img),type(label),type(demo),type(tensor))            
            return img, label, demo, tensor
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)

    
def mimic_csv(df, demo = '../mimic/2.0.0/admissions.csv', demo2 = '../mimic/2.0.0/patients.csv'):
    demo = pd.read_csv(demo)[['subject_id', 'ethnicity']]
    demo2 = pd.read_csv(demo2)[['subject_id', 'gender', 'anchor_age']]
    mimicrace2num = {'AMERICAN INDIAN/ALASKA NATIVE':3,
     'ASIAN':4,
     'BLACK/AFRICAN AMERICAN':2,
     'HISPANIC/LATINO':5,
     'OTHER':0,
     'UNABLE TO OBTAIN':0,
     'UNKNOWN':0,
     'WHITE':1}
    df_t=df.merge(demo.drop_duplicates(), on='subject_id').merge(demo2.drop_duplicates(), on='subject_id')
    df_t['Age'] = df_t['anchor_age']/100
    df_t['Sex'] = df_t['gender'].map(lambda p: {'F':0, 'M':1, 'Unknown':2}[p])
    df_t['ethnicity'] = df_t['ethnicity'].map(lambda p: mimicrace2num[p])
    df_t = df_t[['DicomPath', 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
           'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia',
           'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
           'Fracture', 'Support Devices','ethnicity', 'Age', 'Sex']]
    return df_t

class mimic(data.Dataset):
    def __init__(self, df, root, use_demo = False, train = True, mimicRepetition = None):
        self.datalist = np.array(list(df.itertuples(index=False, name=None)))
        
        self.imgs = [os.path.join(root, k[0]) for k in self.datalist]
        self.labels = df.iloc[:,1:15].copy()
        self.labels = self.labels.replace(-1,1)
        self.labels['competition_no_findings'] = np.array((self.labels['Cardiomegaly'] + 
            self.labels['Edema'] + self.labels['Pleural Effusion'] + 
            self.labels['Atelectasis'] + self.labels['Consolidation'])<1)*1.0
        self.labels = torch.FloatTensor(self.labels.to_numpy())
        self.train = train
        self.repetition = mimicRepetition
        self.use_demo = use_demo
        if use_demo:
            self.demos = (df.iloc[:,[-1,-3,-2]]).to_numpy()
        
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path).convert('RGB')/255
        img = tforms['train'](img) if self.train else tforms['val'](img) 
        # print(img)
        if self.use_demo:
            demo = self.demos[idx]
            #print(demo)
            race = [0,0,0,0,0]
            race[int(demo[2])-1] = 1
            if int(demo[2]) == 0:
                race = [1,1,1,1,1]
            demo = np.concatenate([demo[0:1], race, demo[2:3]])
            
            tensor = labels2tensor(list(label.numpy()) + list(demo), self.repetition)
            
            return img, label, demo, tensor
        else:
            return img, label
    def __len__(self):
        return len(self.imgs)
    
def midrc_csv(df):
    demo = df
    midrcrace2num = {'American Indian or Alaska Native':2,
     'Asian':3,
     'Black or African American':4,
     'Native Hawaiian or other Pacific Islander':5,
     'Other': 0,
     'White': 1}
    demo['Age'] = demo['Age']/100
    demo['Sex'] = demo['Sex'].map(lambda p: {'Female':0, 'Male':1}[p])
    demo['Race'] = demo['Race'].map(lambda p: midrcrace2num[p])
    demo['label'] = demo['label'].map(lambda p: {'No':0, 'Yes':1}[p])
    return demo  
    
    
class midrc(data.Dataset):
    def __init__(self, df, root, train = True, use_demo = True, repetition = None):
        self.datalist = np.array(list(df.itertuples(index=False, name=None)))
        
        self.imgs = [os.path.join(root, k[8]) for k in self.datalist]
        self.labels = df.iloc[:,7].copy()
        self.labels = torch.FloatTensor(self.labels.to_numpy())
        self.train = train
        self.repetition = repetition
        self.use_demo = use_demo
        if use_demo:
            self.demos = (df.iloc[:,[2,3,4]]).to_numpy()
        
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        # img = Image.open(img_path).convert('RGB')/255
        img = tforms['train'](img) if self.train else tforms['val'](img) 
        # print(img)
        if self.use_demo:
            demo = self.demos[idx]
            #print(demo)
            race = [0,0,0,0,0]
            race[int(demo[2])-1] = 1
            if int(demo[2]) == 0:
                race = [1,1,1,1,1]
            demo = np.concatenate([demo[0:1], race, demo[2:3]])
            
            # print(list([label]) + list(demo), self.repetition)
            
            tensor = labels2tensor([label] + list(demo), self.repetition)
            
            return img, label, demo, tensor
        else:
            return img, label
    def __len__(self):
        return len(self.imgs)
    
def save_model(model, epoch, name, task, raw = False):
    # note that this is only deigned for saving during lightninglite training
    if epoch == None:
        file_path = os.path.join('./weights/{}/{}/'.format(task, name), 'model.pt')
    else:
        file_path = os.path.join('./weights/{}/{}/'.format(task, name), 'model-{:05d}.pt'.format(epoch))
        old_path = os.path.join('./weights/{}/{}/'.format(task, name), 'model-{:05d}.pt'.format(epoch-3))
        if os.path.exists(old_path):
            os.remove(old_path)
    Path('./weights/{}/{}/'.format(task, name)).mkdir(parents=True, exist_ok=True)
    # gets model from lightninglite model
    state = (model if raw else [m for m in model.modules()][1]).state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))
    
    

def kl_loss_function(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def matrix_poly(matrix, d, device):
    x = torch.eye(d, device= device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)
    
    
def dag_loss_function(A, m, device):
    expm_A = matrix_poly(A*A, m, device)
    h_A = torch.trace(expm_A) - m
    return h_A

def eval_model(model, model_type, dataset, tasks, verbose = 0, device = None):
    model_previous_status = model.training
    # print('model previous status:', model_previous_status)
    
    model.eval()
    s = nn.Sigmoid()
    
    # print('tasks: ', tasks)

    with torch.no_grad():
        running_auc = 0
        ground_truth = [[] for _ in range(int(sum(tasks)))]
        predictions = [[] for _ in range(int(sum(tasks)))]
        for step, (images, labels) in enumerate(tqdm(dataset)):
            images = images.to(device)
            class_output, _ = predict_batch(model, model_type, images, None, inference = True)
            class_output = s(class_output)
            if tasks.shape[0] > 1:
                labels_sub = labels[:,tasks].cpu().numpy()
                preds_sub = class_output[:,tasks].detach().cpu().numpy()
                for j in range(labels_sub.shape[1]): 
                    label = labels_sub[:,j]
                    pred = preds_sub[:,j]
                    ground_truth[j].append(label)
                    predictions[j].append(pred)
            else:
                ground_truth[0].append(labels.cpu().numpy())
                predictions[0].append(class_output.view_as(labels).cpu().numpy())
                
    ground_truth = [np.concatenate(g) for g in ground_truth]
    predictions = [np.concatenate(p) for p in predictions] 
    
    if verbose == 1:
        return ground_truth, predictions
     
    fpr, tpr, thresholds = metrics.roc_curve(np.concatenate(ground_truth), np.concatenate(predictions), pos_label=1)
    test_auc = metrics.auc(fpr, tpr)
    if model_previous_status:
        model.train()
    
    return test_auc



mimic_path = '../mimic/'
def mimic_age_20(split = 'test'):
    demo = mimic_csv(pd.read_csv(mimic_path + '{}.csv'.format(split)))
    age = demo['Age']
    age = np.array(age)//0.2
    ii = [np.where(age == val)[0] for val in range(0,5)]
    return ii

def mimic_sex(split = 'test'):
    demo = mimic_csv(pd.read_csv(mimic_path + '{}.csv'.format(split)))
    sex = demo['Sex']
    # race = np.array(race)//0.1
    ii = [np.where(sex == val)[0] for val in [0,1]]
    return ii

def mimic_race(split = 'test'):
    demo = mimic_csv(pd.read_csv(mimic_path + '{}.csv'.format(split)))
    race = demo['ethnicity']
    ii = [np.where(race == val)[0] for val in range(0, 6)]
    return ii

def mimic_wnw(split = 'test'):
    demo = mimic_csv(pd.read_csv(mimic_path + '{}.csv'.format(split)))
    race = demo['ethnicity']
    ii = [np.where(race == val)[0] for val in range(0, 6)]
    wnw_list = [ii[1], []]
    wnw_list[1] = list(ii[0]) + list(ii[2]) + list(ii[3]) + list(ii[4]) + list(ii[5])
    return wnw_list

def midrc_age_20(demo = '../MIDRC/split/train_1.csv'):
    demo = pd.read_csv(demo)
    age = demo['Age']/100
    age = np.array(age)//0.2
    ii = [np.where(age == val)[0] for val in range(0,5)]
    return ii

def midrc_sex(demo = '../MIDRC/split/train_1.csv'):
    demo = pd.read_csv(demo)
    sex = demo['Sex'].map(lambda p: {'Female':0, 'Male':1}[p])
    # race = np.array(race)//0.1
    ii = [np.where(sex == val)[0] for val in [0,1]]
    return ii

def midrc_race(demo = '../MIDRC/split/train_1.csv'):
    demo = pd.read_csv(demo)
    # only do this for eval
    midrcrace2num = {'American Indian or Alaska Native':2,
     'Asian':2,
     'Black or African American':3,
     'Native Hawaiian or other Pacific Islander':2,
     'Other': 0,
     'White': 1}
    race = demo['Race'].map(lambda p: midrcrace2num[p])
    ii = [np.where(race == val)[0] for val in range(0, 4)]
    return ii

def midrc_wnw(demo = '../MIDRC/split/train_1.csv'):
    demo = pd.read_csv(demo)
    midrcrace2num = {'American Indian or Alaska Native':2,
     'Asian':3,
     'Black or African American':4,
     'Native Hawaiian or other Pacific Islander':5,
     'Other': 0,
     'White': 1}
    race = demo['Race'].map(lambda p: midrcrace2num[p])
    ii = [np.where(race == val)[0] for val in range(0, 6)]
    wnw_list = [ii[1], []]
    wnw_list[1] = list(ii[0]) + list(ii[2]) + list(ii[3]) + list(ii[4]) + list(ii[5])
    return wnw_list

def chexpert_race(path):
    demo = pd.read_csv(path)
    race = demo['PRIMARY_RACE']
    race = np.array(race)
    ii = [np.where(race == val)[0] for val in range(0, 6)]
    ii[4] = ii[5]
    ii = ii[:5]
    return ii

def chexpert_age(path):
    demo = pd.read_csv(path)
    age = demo['Age']
    age = np.array(age)//0.2
    ii = [np.where(age == val)[0] for val in range(0,5)]
    return ii

def chexpert_sex(path):
    demo = pd.read_csv(path)
    sex = demo['Sex']
    # race = np.array(race)//0.1
    ii = [np.where(sex == val)[0] for val in [0,1]]
    return ii
