from networks.deeplab_resnet import Res_Deeplab
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
    label = label
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
parser.add_argument('--timesteps', metavar='timesteps', type=int, nargs=1, default=14,
                    help='Number of timesteps (frames) to train RNN on')
parser.add_argument('--iters', metavar='iterations', type=int, nargs=1, default=30000,
                    help='Number of iterations to train')
parser.add_argument('--save_step', metavar='savestep', type=int, nargs=1, default=100,
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
vismem.load_state_dict(torch.load("data/models/vismem_80.pth"))

if args.cuda_vismem: vismem.cuda()

deep_lab = Res_Deeplab()
#deep_lab.load_state_dict( torch.load("data/models/MS_DeepLab_resnet_pretrained_COCO_init.pth"))
deep_lab.load_state_dict( torch.load("data/models/bigmem2/deep_lab_4100.pth"))
b4 = deep_lab.state_dict()

if args.cuda_deeplab: deep_lab.cuda()
l8r = deep_lab.state_dict()

database = Database(args.DAVIS_base, args.image_set)
base_lr = 1e-4
optimizer = optim.RMSprop(vismem.parameters(), lr=1e-4, weight_decay=0.005)
logsoftmax = nn.LogSoftmax()
for i in range(0, args.iters+1):
    overall = time.time()
    data_read = time.time()
    images, targets = database.get_next(args.timesteps+1)
    data_read = time.time()-data_read
    opt_zero = time.time()
    optimizer.zero_grad()
    opt_zero = time.time()-opt_zero
    rescale = nn.UpsamplingBilinear2d(size = ( images[0].shape[2], images[0].shape[3] )).cuda()

    appearance_tt = 0
    vism_t = 0
    loss_tt = 0
    cuda_tt = 0

    for s in range(args.timesteps):
        image = Variable(torch.Tensor.pin_memory(torch.from_numpy(images[s]).float()), volatile=True)
        mask = Variable(torch.Tensor.pin_memory(torch.from_numpy(targets[s]).float()))

        cuda_t = time.time()
        if args.cuda_deeplab: image = image.cuda()
        mask = mask.cuda()
        cuda_t = time.time()-cuda_t


        appearance_t = time.time()
        appearance = deep_lab(image)[3]
        appearance = Variable(appearance.data)
        appearance = appearance.cuda()
        appearance_t = time.time()-appearance_t
        #appearance.volatile = False
        #appearance.requires_grad = True
        #print([(a[0][0],a[0][1].cpu()-a[1][1].cpu()) for a in zip(b4.items(), l8r.items())])
        #if args.cuda_vismem:

        if s == 0:
            h_next = None
            mask_pred = mask
            mask_prev = mask
        else:
            #if random.random() < 0.2:
            #    mask_pred = mask_prev
            #else:
            mask_pred = (math.e**logsoftmax(mask_pred))[:, 1:2, :, :]
            #mask_prev = mask
            #mask_pred = mask_pred[:, 1:2, :, :]
        vism = time.time()
        mask_pred, h_next = vismem(appearance, mask_pred, h_next)
        vism = time.time()-vism

        loss_t = time.time()
        if s == 0:
            loss = loss_calc(rescale(mask_pred), targets[s], args.cuda_vismem)
        else:
            loss = loss + loss_calc(rescale(mask_pred), targets[s], args.cuda_vismem)
        loss_t = time.time()-loss_t
        appearance_tt = appearance_tt + appearance_t
        vism_t = vism_t + vism
        loss_tt = loss_tt + loss_t
        cuda_tt = cuda_tt + cuda_t

    #dot = make_dot(mask_pred)
    #dot.render("data/graphs/{}.gv".format(i), view=True)
    overall = time.time()-overall
    lr_ = lr_poly(base_lr,float(i),float(args.iters),0.9)
    rest = overall - appearance_tt - vism_t - loss_tt - data_read - opt_zero - cuda_t
    print("{} Iter: {} Loss: {:.4f}, lr {:.4f}, overall {:.4f}, appearance {:.4f}, vism {:.4f}, loss {:.4f}, opt_zero {:.4f}, data load {:.4f}, rest {:.4f}, cuda {:.4f}".format(datetime.now(), i, loss.data[0], lr_, overall, appearance_tt,
                                        vism_t, loss_tt, opt_zero, data_read, rest, cuda_tt ))
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
       optimizer = optim.RMSprop(vismem.parameters(), lr=lr_, weight_decay=0.005)

    if i % display_step == 0:
        pass
        #for p in preds:
        #    plt.imshow(p[0][1].cpu().numpy(), cmap='gray')
        #    plt.show()
    if i % args.save_step == 0:
        torch.save(vismem.state_dict(),'data/models/bigmem2/vismem_'+str(i)+'.pth')
        torch.save(deep_lab.state_dict(),'data/models/bigmem2/deep_lab_'+str(i)+'.pth')

    #print([(a[0][0],a[1][0]) for a in zip(b4.items(), l8r.items())])
