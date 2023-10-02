import pandas as pd
import re
import skimage, torchvision
import numpy as np
import sys
import warnings
import os
import math
import re
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
from sklearn import preprocessing, utils
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
from pytorch_lightning.lite import LightningLite
from RES_VAE import *
from models import *
from utils import *
import warnings
warnings.filterwarnings("ignore")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', type=int, default=25)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--class_beta', type=float, default=1.2)
    parser.add_argument('--logging_step', type=int, default=400)
    parser.add_argument('--model_name', type=str, default="test_model")
    parser.add_argument('--type', type=str, default='CausalVAE')
    parser.add_argument('--root', type=str, default='../chexpertchestxrays-u20210408/')
    parser.add_argument('--repetition', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default = 8)
    parser.add_argument('--dis_beta', type=float, default = 0.3)
    parser.add_argument('--task', type=str, default = 'chexpert')
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--dagloss', type=int, default = 0)
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

image_size = args.image_size
batch_size = args.batch_size
epoch_num = args.epoch_num
root = args.root

model_type = args.type
logging_step = args.logging_step
class_beta = args.class_beta

# competition_tasks = torch.ByteTensor([0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1]) # one extra for no finding in competition tasks
competition_tasks = torch.ByteTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
# repetition7 = [32 if i < 15 else (6 if (i == (15 + 7)-1 or i == 15 )else 4) for i in range(0, (15 + 7))]
repetition7 = [30 if i < 15 else [9,9,9,9,9,9,8][i-15] for i in range(0, (15 + 7))]


if args.task == 'chexpert':


    repetition = repetition7 #if ("densenet" not in model_type) else chexpertDensenet201_7


    df = pd.read_csv(os.path.join(root, './CheXpert-v1.0/train.csv'))
    df = prepare_cxr_dataset(root, 'CHEXPERT DEMO.xlsx', df)


    train_dataset, valid_dataset = train_test_split( df, test_size=2000)
    valid_path = 'temp_{}.csv'.format(args.model_name)
    valid_dataset.to_csv(valid_path)

    train_dataset = chexpert(train_dataset, root, use_demo = 'Causal' in model_type, chexpertRepetition = repetition)
    train_dataset = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = args.num_workers)
    valid_dataset = chexpert(valid_dataset, root, use_demo = False, chexpertRepetition = repetition)
    valid_dataset = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers = args.num_workers)
    
elif args.task == 'mimic':

    repetition = repetition7
    train_dataset = mimic_csv(pd.read_csv('../mimic/train.csv'))
    valid_dataset = mimic_csv(pd.read_csv('../mimic/validate.csv'))
    train_dataset = mimic(train_dataset, root = root, use_demo = 'Causal' in model_type, mimicRepetition = repetition7 )
    train_dataset = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = args.num_workers)
    valid_dataset = mimic(valid_dataset, root = root, use_demo = False, mimicRepetition = repetition7 )
    valid_dataset = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers = args.num_workers)
    
elif args.task == 'MIDRC':
#     '../MIDRC/data-jpg'
    competition_tasks = torch.ByteTensor([1])
    repetition = [276 + 180, 8, 8,8,8,8,8,8]
    
    index = re.findall(r'\d+', args.model_name)[0]
    
    train_path = '../MIDRC/split/train_{}.csv'.format(index)
    valid_path = '../MIDRC/split/valid_{}.csv'.format(index)
    
    print(train_path)
    
    print(valid_path)
    file_mapping = pd.read_csv('../MIDRC/filtered_final2.csv')[['Path', 'filename']]
    train_dataset = midrc_csv(pd.read_csv(train_path))
    train_dataset = pd.merge(train_dataset,file_mapping[['Path', 'filename']], on = 'Path')
    valid_dataset = midrc_csv(pd.read_csv(valid_path))
    valid_dataset = pd.merge(valid_dataset,file_mapping[['Path', 'filename']], on = 'Path')
    train_dataset = midrc(train_dataset, root = root, use_demo = 'Causal' in model_type, repetition = repetition )
    train_dataset = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = args.num_workers)
    valid_dataset = midrc(valid_dataset, root = root, use_demo = False, repetition = repetition )
    valid_dataset = Data.DataLoader(valid_dataset, batch_size=batch_size*2, shuffle=False, num_workers = args.num_workers)

print('Training length: ', len(train_dataset))
print('Validation length: ', len(valid_dataset))
    


model, vae = load_model(model_type, repetition, out_size = 1 if args.task == 'MIDRC' else 15) if args.newload else load_model_old(model_type, repetition)

if model != None: 
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
head_loss_fn = nn.BCEWithLogitsLoss()

if vae != None:
    vae.to(device)
    optimizer_vae = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()

logging_condition = lambda step, c1, c2: (step % c1 == 0) or (step == c2)

disentangle_beta = args.dis_beta

class Lite(LightningLite):
    def run(self, model, vae, model_type, train_dataset, valid_dataset, optimizer, repetition):
        if args.task == 'mimic':
            r_ii, a_ii, s_ii = mimic_race('validate'), mimic_age_20('validate'), mimic_sex('validate')
        elif args.task == 'MIDRC':                                                                               
            r_ii, a_ii, s_ii = midrc_race(valid_path), midrc_age_20(valid_path), midrc_sex(valid_path)
        elif args.task == 'chexpert':     
            r_ii, a_ii, s_ii = chexpert_race(valid_path), chexpert_age(valid_path), chexpert_sex(valid_path)
        else:
            exception('undefined task')
        bcelogit = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss(reduction = 'mean')
        l1 = nn.L1Loss()
        s = nn.Sigmoid()
        mask = None

        best_auc = 0

        improved = False
        no_improvement = 0
        
        model, optimizer = self.setup(model, optimizer)
        train_dataset = self.setup_dataloaders(train_dataset)
        valid_dataset = self.setup_dataloaders(valid_dataset)

        for epoch in range(0, epoch_num):
            improved = False
            save_model(model, epoch,  args.model_name, args.task)
            for step, datapack in enumerate(tqdm(train_dataset)):
            # for step, datapack in enumerate(train_dataset):
                model.train()
                logging_flag = logging_condition(step, logging_step, len(train_dataset) - 1) # and step!= 0
                logging_content = {}
                if 'Causal' in model_type:
                    images, labels, demos, label_tensor = datapack
                    label_tensor = label_tensor.float()
                else:
                    images, labels = datapack
                    label_tensor = None
                images = images.to(self.device)
                with torch.cuda.amp.autocast():
                    result = predict_batch(model, model_type, images, vae, label_tensor = label_tensor, discriminator = None, task = args.task)
                    loss = 0

                    if 'VAE' in model_type:
                        kl_loss = kl_loss_function(result['mu'], result['var'])
                        mse_loss = F.mse_loss(result['recon_img'], images)
                        loss += args.beta * kl_loss + mse_loss
                        logging_content['vae loss'] = float(loss)

                    # disease loss


                    class_output = result['disease_classes'] 
                    # print(class_output)
                    class_loss = class_beta * bcelogit(class_output.view_as(labels), labels)
                    logging_content['class_loss'] = float(class_loss)
                    loss += class_loss

                    # protected loss
                    if 'Causal' in model_type:
                        demos = demos.float().to(self.device)

                        SuLoss = bcelogit(result['causal_u'][:, :repetition[1]], label_tensor[:, :repetition[1]])
                        RuLoss = bcelogit(result['causal_u'][:, repetition[1]:-repetition[-1]], label_tensor[:, repetition[1]:-repetition[-1]])
                        AuLoss = l1(result['causal_u'][:, -repetition[-1]:], label_tensor[:, -repetition[-1]:])
                        
                        logging_content['SuLoss'] = float(SuLoss)
                        logging_content['RuLoss'] = float(RuLoss)
                        logging_content['AuLoss'] = float(AuLoss)
                        if args.dagloss:
                            dag_param = result['dag']
                            if mask == None:
                                mask = torch.ones(dag_param.size()).to(self.device) - torch.eye(dag_param.size()[0]).to(self.device)
                                mask.requires_grad = False

                            dag_param = dag_param * mask
                            h_a = dag_loss_function(dag_param, dag_param.size()[0], self.device)
                            dag_loss = 3*h_a + 0.5*h_a*h_a 
                            logging_content['dag loss'] = float(dag_loss)
                            loss += dag_loss
                            logging_content['sum graph'] = torch.norm(result['dag'])
                        
                        loss += SuLoss + RuLoss + AuLoss
                    

                    optimizer.zero_grad()
                    self.backward(loss, retain_graph=True)
                    if model != None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    if vae != None:
                        torch.nn.utils.clip_grad_norm_(vae.parameters(), 5)
                    optimizer.step()

              
                        
                    if logging_flag:
                        logging_content['Total loss'] = float(loss)
                       
                        ground_truth, predictions = eval_model(model, model_type, valid_dataset, competition_tasks, verbose = 1)
                        step_auc = metrics.roc_auc_score(np.array(ground_truth).T, np.array(predictions).T, multi_class =  'ovr', average = 'macro')
                        print('Task AUC: ', step_auc)
                        demo_performance = []

                        for demo in [r_ii, s_ii, a_ii]:
                            
                            if args.task != 'MIDRC':
                                demo = [np.concatenate([index_array + n*len(predictions[0]) for n in range(0, 14)]) for index_array in demo]
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
                                other_logits = other_logits.reshape(1,-1)
                                average_matrix = [np.mean(np.where(a-other_logits.reshape(1,-1)>0,1,0)) for a in pos_logit_current]
                                group_dependent_acc.append(np.mean(average_matrix))
                                

                            group_dependent_acc = [100 * v for v in group_dependent_acc]
                            demo_performance.append(group_dependent_acc)
                        demo_performance = [np.std(v) for v in demo_performance]
                        
                        logging_content['demo'] = demo_performance
                                 
                 
                        
                        
                        print(logging_content)
                        
                        if step_auc > best_auc:
                            improved = True
                            best_model = copy.deepcopy(model.state_dict())
                            model.load_state_dict(best_model)
                            model = model.to(self.device)
                            best_auc = step_auc

            model.load_state_dict(best_model)
            model = model.to(self.device)

            if model_type != 'VAE':
                save_model(model, epoch,  args.model_name, args.task)
            else:
                save_model(vae, epoch, 'VAE')


            if not improved:
                print('no improvement')
                no_improvement += 1
                if no_improvement >= 3:
                    print('no improvement for 3 epochs')
                    return model#, vae
            else:
                print('improved epoch/still needs training')
                print('prev', best_auc, 'current ', best_race)
                no_improvement = 0

        return model


# if model_type == 'vae':
#     _, vae = train(None, vae, model_type, train_dataset, optimizer_vae)
# else:
model = Lite(devices=[args.gpu], accelerator="auto", precision=16).run(model, vae, model_type, train_dataset, valid_dataset, optimizer_model, repetition)
save_model(model, None,  args.model_name, args.task)
