import pandas as pd
import re
import skimage, torchvision
import numpy as np
import sys
import warnings
import os
import math
from PIL import Image
import argparse
import time
import cv2
import copy
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import autograd
from torch.autograd import Variable, Function
from torch.utils import data
import torchvision
from RES_VAE import VAE
# from utils import grad_reverse

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class CausalDensenet(nn.Module):
    def __init__(self, out_size = 15, nnIsTrained = True, causal = False, repetition = None, convDepth = 1):
        super(CausalDensenet, self).__init__()
        self.densenet = torchvision.models.densenet201(pretrained = nnIsTrained)
        num_ftrs = self.densenet.classifier.in_features
        self.repetition = repetition
        self.densenet.classifier = nn.Linear(in_features=num_ftrs, out_features=sum(self.repetition), bias=True)
        self.disease_classifier = classifier_head(in_size = sum(self.repetition), out_size = out_size)
        self.global_noise = {}
        # densenet image component is features, classifier is classifiers 
        
        self.causal = causal
        if self.causal:
            self.nonlinearMasku = nn.ModuleList([nn.Sequential(nn.Linear(rep , 128),
                            nn.ELU(),
                            nn.Linear(128 , rep),
                           ) 
                          for rep in self.repetition])
            self.nonlinearMaskz = nn.ModuleList([nn.Sequential(nn.Linear(rep , 128),
                                nn.ELU(),
                                nn.Linear(128 , rep),
                               ) 
                          for rep in self.repetition])
            self.graph = DependencyGraph(sum(self.repetition), depth = convDepth)
            
    def load_lightning(self, weights):
        self.densenet = weights.densenet
        self.repetition = weights.repetition
        self.disease_classifier = weights.disease_classifier
        self.causal = weights.causal
        if self.causal:
            self.graph.A = weights.dag.A
            self.nonlinearMasku = weights.nonlinearMasku
            self.nonlinearMaskz = weights.nonlinearMaskz
        
        

    def forward(self, x, u, num_target = 15):
        # x = self.densenet121(x)
        result = {}
        z = self.densenet(x)
        # print(z.shape)
        
        if self.causal:
            epsilon = torch.squeeze(z)
            if len(epsilon.shape) == 1:
                epsilon = epsilon.view([1, -1])
            z_decomp = self.graph.transform_SEM(epsilon)
            z = self.graph(z_decomp) #+ epsilon
            # mask Z to approximate causal effect
            pointer = 0
            nonlinearZ = []
            
            for rep, f in zip(self.repetition, self.nonlinearMaskz):
                pointer += rep
                nonlinearZ.append(f(z[:, pointer-rep:pointer]))
            z = torch.cat(nonlinearZ, dim = 1) + epsilon
            if self.training:
                # mask u to approximate causal effect in order to assist learning of causal graph
                u = u.float()
                u = self.graph(u) 
                nonlinearU = []
                pointer = 0
                for rep, f in zip(self.repetition, self.nonlinearMasku):
                    pointer += rep
                    nonlinearU.append(f(u[:, pointer-rep:pointer]))
                causal_u = torch.cat(nonlinearU, dim = 1)

                result['causal_u'] = causal_u
                
                # randomly permute / add noise
                # permute:
                # len_target_dim = sum(self.repetition[:num_target])
                # z = torch.cat([z[:,:len_target_dim],z[:,len_target_dim:][torch.randperm(z.size()[0])]], dim = 1)
                
                # add noise:
                
                
                
                disease_classes = []
                for noise_iter in range(10):
                    pointer = 0
                    noisy_z = []
                    for rep, itr in zip(self.repetition, range(len(self.repetition))):
                        pointer += rep
                        sub_z = z[:,pointer-rep:pointer]
                        if itr >= num_target:
                            # # update only once per batch
                            # if noise_iter == 0:
                            v = torch.var(sub_z, dim = 0)
                            if self.global_noise.get(itr) == None:
                                self.global_noise[itr] = v.detach().clone()
                            else:
                                v_temp = v.detach().clone()
                                # print(self.global_noise[itr].shape, v,sub_z, rep, pointer, v_temp.shape)
                                self.global_noise[itr] = (2*self.global_noise[itr] + v_temp)
                                self.global_noise[itr] = self.global_noise[itr] /3
                            r = torch.randn_like(sub_z)
                            sub_z = r*self.global_noise[itr] + sub_z
                        noisy_z.append(sub_z)
                    # print(len(noisy_z))
                    z_ = torch.cat(noisy_z, dim = 1)
                    disease_classes.append(self.disease_classifier(z_))
                
                
                
            result['dag'] = self.graph.get_A()
        result['z'] = z
        
        if self.training:
            result['disease_classes'] = sum(disease_classes)/len(disease_classes)
        else:
            result['disease_classes'] = self.disease_classifier(z)
            
        return result


class CausalVAEclassifier(nn.Module):
    def __init__(self, causal = False, disentangle = False, adversarial = False, repetition = None, before = False, seperate = False, convDepth = 1, out_size = 15):
        super(CausalVAEclassifier, self).__init__()
        self.vae = VAE(3)
        self.causal = causal
        self.disentangle = disentangle
        self.adversarial = adversarial
        self.before = before
        self.seperate = seperate
        self.repetition = repetition
        self.global_noise = {}
        if self.causal:
            self.nonlinearMasku = nn.ModuleList([nn.Sequential(nn.Linear(rep , 128),
                            nn.ELU(),
                            nn.Linear(128 , rep),
                           ) 
                          for rep in self.repetition])
            self.nonlinearMaskz = nn.ModuleList([nn.Sequential(nn.Linear(rep , 128),
                                nn.ELU(),
                                nn.Linear(128 , rep),
                               ) 
                          for rep in self.repetition])
            self.graph = DependencyGraph(sum(self.repetition), depth = convDepth)
        self.disease_classifier = classifier_head(in_size = sum(self.repetition), out_size = out_size)
        # print(self.disease_classifier)

        if self.disentangle:
            self.disease_pos = classifier_head(in_size = sum(self.repetition[0:15]), out_size = out_size)
            
    def load_lightning(self, weights):
        self.vae = weights.vae
        self.causal = weights.causal
        self.disentangle = weights.disentangle
        self.adversarial = weights.adversarial
        self.before = weights.before
        self.seperate = weights.seperate
        self.repetition = weights.repetition
        self.disease_classifier = weights.disease_classifier
        
        if self.causal:
            self.graph.A = weights.dag.A
            self.nonlinearMasku = weights.nonlinearMasku
            self.nonlinearMaskz = weights.nonlinearMaskz
        if self.disentangle:
            self.disease_pos = weights.disease_pos
  
        
    def forward(self, images, u, discriminator = None, conv = 2, num_target = 15):
        causal_u = None
        result = {}
        encoding, mu, var = self.vae.encoder(images)
        result['mu'] = mu
        result['var'] = var
        z = encoding
        
        if self.causal:
            epsilon = torch.squeeze(encoding)
            if len(epsilon.shape) == 1:
                epsilon = epsilon.view([1, -1])
            z = self.graph.transform_SEM(epsilon)
            z = self.graph(z)
            
            pointer = 0
            nonlinearZ = []
            for rep, f in zip(self.repetition, self.nonlinearMaskz):
                pointer += rep
                nonlinearZ.append(f(z[:, pointer-rep:pointer]))
            z = torch.cat(nonlinearZ, dim = 1) + epsilon
            if self.training:
                u = self.graph(u.float())
                nonlinearU = []
                pointer = 0
                for rep, f in zip(self.repetition, self.nonlinearMasku):
                    pointer += rep
                    nonlinearU.append(f(u[:, pointer-rep:pointer]))
                causal_u = torch.cat(nonlinearU, dim = 1)

                result['causal_u'] = causal_u
                
                # add noise:
                pointer = 0
                noisy_z = []
                
                disease_classes = []
                for noise_iter in range(10):
                    pointer = 0
                    noisy_z = []
                    for rep, itr in zip(self.repetition, range(len(self.repetition))):
                        pointer += rep
                        sub_z = z[:,pointer-rep:pointer]
                        if itr >= num_target:
                            # # update only once per batch
                            # if noise_iter == 0:
                            v = torch.var(sub_z, dim = 0)
                            if self.global_noise.get(itr) == None:
                                self.global_noise[itr] = v.detach().clone()
                            else:
                                v_temp = v.detach().clone()
                                # print(self.global_noise[itr].shape, v,sub_z, rep, pointer, v_temp.shape)
                                self.global_noise[itr] = (2*self.global_noise[itr] + v_temp)
                                self.global_noise[itr] = self.global_noise[itr] /3
                            r = torch.randn_like(sub_z)
                            sub_z = r*self.global_noise[itr] + sub_z
                        noisy_z.append(sub_z)
                    # print(len(noisy_z))
                    z_ = torch.cat(noisy_z, dim = 1)
                    disease_classes.append(self.disease_classifier(z_))
        
            result['dag'] = self.graph.A
        # generate image
        
        recon_img = self.vae.decoder(z.view([z.shape[0], z.shape[1], 1, 1]))
        result['recon_img'] = recon_img


        if self.training:
            result['disease_classes'] = sum(disease_classes)/len(disease_classes)
        else:
            result['disease_classes'] = self.disease_classifier(torch.squeeze(z))

        result['z'] = z
        return result
    