from networks.deeplab_masktrack import Deeplab_Masktrack
from networks.vismem import VisMem
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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def loss_calc(out, label, cuda):
    """
    This function returns cross entropy loss for segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape batch_size x 1 x h x w  -> batch_size x h x w
    label = label+127.5
    label[label==255] = 1
    label = torch.from_numpy(label).long()
    label = label[:, 0, :, :]
    label = Variable(label)
    label = label.cuda()
    out  = out.cuda()
    m = nn.LogSoftmax().cuda()
    criterion = nn.NLLLoss2d().cuda()
    #criterion = nn.BCELoss()
    out = m(out)
    return criterion(out,label)

def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))

parser = argparse.ArgumentParser(description='Train the vismem network')
parser.add_argument('--timesteps', metavar='timesteps', type=int, nargs=1, default=2,
                    help='Number of timesteps (frames) to train RNN on')
parser.add_argument('--iters', metavar='iterations', type=int, nargs=1, default=300000,
                    help='Number of iterations to train')
parser.add_argument('--save_step', metavar='savestep', type=int, nargs=1, default=10000,
                    help='Number of iterations between saves')
parser.add_argument('--update_step', metavar='update_step', type=int, nargs=1, default=10,
                    help='Number of iterations between saves')
parser.add_argument('--cuda_vismem', metavar='cuda_vismem', type=bool, nargs=1, default=True,
                    help='True if vismem should be run on cuda cores')
parser.add_argument('--cuda_deeplab', metavar='cuda_deeplab', type=bool, nargs=1, default=True,
                    help='True if deeplab should be run on cuda cores')
parser.add_argument('--DAVIS_base', metavar='DAVIS_base', type=str, nargs=1, default="data/DAVIS",
                    help='Location of DAVIS')
parser.add_argument('--image_set', metavar='image_set', type=str, nargs=1, default="data/DAVIS/ImageSets/480p/train.txt",
                    help='Location of the list of training pairs')
args = parser.parse_args()
display_step = 10
gpu=3

vismem = VisMem(2048,128,128,7,args.cuda_vismem)
#vismem.load_state_dict(torch.load("data/models/mtv/vismem_13000.pth"))

if args.cuda_vismem: vismem.cuda()

deep_lab = Deeplab_Masktrack()
#deep_lab.load_state_dict( torch.load("data/models/MS_DeepLab_resnet_pretrained_COCO_init.pth"))
deep_lab.load_state_dict( torch.load("data/models/masktrack_v22/masktrack_30000.pth"))

if args.cuda_deeplab: deep_lab.cuda()

database = Database(args.DAVIS_base, args.image_set)
base_lr = 1e-4
lr_ = base_lr
optimizer = optim.RMSprop(vismem.parameters(), lr=base_lr, weight_decay=0.005)
logsoftmax = nn.LogSoftmax()
last_ten = []
optimizer.zero_grad()

for i in range(0, args.iters+1):
    if i>60000:
       args.updat_step=1
       args.timesteps=((i-60000)/1000)+3
    overall = time.time()
    data_read = time.time()
    images, targets = database.get_next(args.timesteps+1)
    data_read = time.time()-data_read
    rescale = nn.UpsamplingBilinear2d(size = ( images[0].shape[2], images[0].shape[3] )).cuda()

    overall_t = time.time()
    mask_pred = torch.from_numpy(targets[0]).float().cuda()
    h_next = None
    for s in range(1, args.timesteps):
        if s>1:
           mask_pred = ((math.e**logsoftmax(mask_pred.data))[:, 1:2, :, :]-0.5)*255
           mask_pred = mask_pred.data

        image = torch.cat((torch.from_numpy(images[s]).cuda(), mask_pred), dim=1)
        image = Variable(image, volatile=True).float()
        mask = Variable(torch.from_numpy(targets[s]).float())
        if args.cuda_deeplab: image = image.cuda()
        mask = mask.cuda()

        appearance = deep_lab(image)[3]
        appearance = Variable(appearance.data)
        appearance = appearance.cuda()

        mask_pred, h_next = vismem(appearance, mask_pred, h_next)
        mask_pred = rescale(mask_pred)

        if s == 1:
            loss = loss_calc(mask_pred, targets[s], args.cuda_vismem)
        else:
            loss = loss + loss_calc(mask_pred, targets[s], args.cuda_vismem)


    last_ten.append(loss.data[0])
    if len(last_ten)>10: last_ten.pop(0)
    overall_t = time.time() - overall_t
    
    print("{} Iter: {} Loss: {:.4f}, lr {:.4f}, overall {:.4f}, aveloss: {:.4f}, timesteps: {}".format(datetime.now(), i, loss.data[0], lr_, overall_t, sum(last_ten)/len(last_ten), args.timesteps ))
    if i % args.update_step == 0:
       loss=loss/args.update_step
       lr_ = lr_poly(base_lr,float(i),float(args.iters),0.9)
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

    if i % 1000 == 0:
        param_groups = optimizer.param_groups
        param_groups[0]['lr'] = lr_
        #optimizer = optim.RMSprop(vismem.parameters(), lr=lr_, weight_decay=0.005)

    if i % display_step == 0:
        pass
        #for p in preds:
        #    plt.imshow(p[0][1].cpu().numpy(), cmap='gray')
        #    plt.show()
    if i % args.save_step == 0:
        torch.save(vismem.state_dict(),'data/models/mtv3/vismem_'+str(i)+'.pth')
