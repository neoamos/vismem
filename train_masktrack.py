from networks.deeplab_masktrack import Deeplab_Masktrack
from database import Database
from datetime import datetime
import numpy as np
import sys
import torch
import os
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as f
import math
import time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def loss_calc(out, label, cuda):
    """
    This function returns cross entropy loss for segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape batch_size x 1 x h x w  -> batch_size x h x w
    rescale = nn.UpsamplingBilinear2d(size = ( out.size()[2], out.size()[3] )).cuda()

    label = label+127.5
    label[label==255] = 1
    label = torch.from_numpy(label).float()[:, 0, :, :]
    label = Variable(label).long()
    #label = rescale(label).data[:, 0, :, :]
    #label = Variable(label.long())
    if cuda: label = label.cuda()
    if cuda: out  = out.cuda()
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()
    #criterion = nn.BCELoss()

    out = m(out)
    return criterion(out,label)

def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())
    #b.append(model.Scale.layer6.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))

parser = argparse.ArgumentParser(description='Train the vismem network')
parser.add_argument('--iters', metavar='iterations', type=int, nargs=1, default=30000,
                    help='Number of iterations to train')
parser.add_argument('--save_step', metavar='savestep', type=int, nargs=1, default=1000,
                    help='Number of iterations between saves')
parser.add_argument('--cuda', metavar='cuda', type=bool, nargs=1, default=True,
                    help='True if model should be run on cuda cores')
parser.add_argument('--DAVIS_base', metavar='DAVIS_base', type=str, nargs=1, default="data/DAVIS",
                    help='Location of DAVIS')
parser.add_argument('--image_set', metavar='image_set', type=str, nargs=1, default="data/DAVIS/ImageSets/480p/train.txt",
                    help='Location of the list of training pairs')
parser.add_argument('--batch_size', metavar='batch_size', type=str, nargs=1, default=6,
                    help='batch size for training')
args = parser.parse_args()
display_step = 10

#ewdim = torch.rand((64,1,7,7))
#print(weights['Scale.conv1.weight'].shape)
#print(type(weights))
#weights['Scale.conv1.weight'] = torch.cat((weights['Scale.conv1.weight'], newdim), dim=1)
#weights.pop('Scale.layer5.conv2d_list.0.weight', None)
#for k in weights.keys():
#    if "layer5" in k:
#        weights.pop(k, None)
#print([(k, weights[k].shape) for k in weights.keys()])

deep_lab = Deeplab_Masktrack()
pretrained_dict = torch.load("data/models/MS_DeepLab_resnet_pretrained_COCO_init.pth")
model_dict = deep_lab.state_dict()
newdim = model_dict['Scale.conv1.weight'][:, 0:1, :, :]
pretrained_dict['Scale.conv1.weight'] = torch.cat((pretrained_dict['Scale.conv1.weight'], newdim), dim=1)
#print(pretrained_dict.keys())
for k in pretrained_dict.keys():
   if "layer5" in k: pretrained_dict.pop(k, None)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)

deep_lab.load_state_dict(model_dict)

if args.cuda: deep_lab.cuda()

database = Database(args.DAVIS_base, args.image_set)
base_lr = 1e-3
lr_ = base_lr
weight_decay = 0.0005
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(deep_lab), 'lr': base_lr },
                        {'params': get_10x_lr_params(deep_lab), 'lr': base_lr*10} ],
                        lr = base_lr, momentum = 0.9,weight_decay = weight_decay)
optimizer.zero_grad()
last_ten = []
for i in range(0, args.iters):
    overall_t = time.time()
    sources, targets = database.get_next_masktrack(args.batch_size)
    rescale = nn.UpsamplingBilinear2d(size = ( sources.shape[2], sources.shape[3])).cuda()
    rescale2 = nn.UpsamplingBilinear2d(size = ( 321, 321)).cuda()

    sources = Variable(torch.from_numpy(sources).float())
    sources = rescale2(sources)
    if args.cuda: sources = sources.cuda()
    out = deep_lab(sources)
    loss = loss_calc(rescale(out[0]), targets, args.cuda)
    for j in range(1, len(out)):
        loss = loss + loss_calc(rescale(out[j]), targets, args.cuda)
    loss = loss/args.batch_size
    loss.backward()
    overall_t = time.time() - overall_t

    last_ten.append(loss.data[0])
    if len(last_ten)>10: last_ten.pop(0)
    print("{} Iter: {} Loss: {:.4f}, ave_loss {:0.4f}, lr {:.7f}, time {:0.4f}".format(datetime.now(), i, loss.data[0], sum(last_ten)/len(last_ten), lr_, overall_t))

    if i % 1 == 0:
       optimizer.step()
       lr_ = lr_poly(base_lr,float(i),float(args.iters),0.9)
       param_groups = optimizer.param_groups
       param_groups[0]['lr'] = lr_
       param_groups[1]['lr'] = lr_*10
       optimizer.zero_grad()

    if i % args.save_step == 0:
        torch.save(deep_lab.state_dict(),'data/models/masktrack_invalgoodbatch/masktrack_'+str(i)+'.pth')
