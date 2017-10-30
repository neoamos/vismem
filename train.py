from networks.deeplab_resnet import Res_Deeplab
from networks.vismem import VisMem
from database import Database
from datetime import datetime
from graphviz import Digraph
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
    if cuda: label = label.cuda()

    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()
    #criterion = nn.BCELoss()

    out = m(out)
    return criterion(out,label)
def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

parser = argparse.ArgumentParser(description='Train the vismem network')
parser.add_argument('--timesteps', metavar='timesteps', type=int, nargs=1, default=3,
                    help='Number of timesteps (frames) to train RNN on')
parser.add_argument('--iters', metavar='iterations', type=int, nargs=1, default=5000,
                    help='Number of iterations to train')
parser.add_argument('--save_step', metavar='savestep', type=int, nargs=1, default=10,
                    help='Number of iterations between saves')
parser.add_argument('--cuda_vismem', metavar='cuda_vismem', type=bool, nargs=1, default=True,
                    help='True if vismem should be run on cuda cores')
parser.add_argument('--cuda_deeplab', metavar='cuda_deeplab', type=bool, nargs=1, default=False,
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
deep_lab.load_state_dict( torch.load("data/models/MS_DeepLab_resnet_pretrained_COCO_init.pth"))
if args.cuda_deeplab: deep_lab.cuda()

database = Database(args.DAVIS_base, args.image_set)

optimizer = optim.RMSprop(vismem.parameters(), lr=1e-4, weight_decay=0.005)
logsoftmax = nn.LogSoftmax()
for i in range(60, args.iters):
    images, targets = database.get_next(args.timesteps+1)
    optimizer.zero_grad()
    rescale = nn.UpsamplingBilinear2d(size = ( images[0].shape[2], images[0].shape[3] ))

    preds = []
    for s in range(args.timesteps):
        image = Variable(torch.from_numpy(images[s]).float(), requires_grad=False)
        mask = Variable(torch.from_numpy(targets[s])).float()

        if args.cuda_deeplab: image = image.cuda()
        appearance = deep_lab(image)[3]
        appearance = Variable(appearance.data)
        if args.cuda_vismem:
            appearance = appearance.cuda()
            mask = mask.cuda()

        if s == 0:
            h_next = None
            mask_pred = mask
        else:
            mask_pred = (math.e**logsoftmax(mask_pred))[:, 1:2, :, :]
            #mask_pred = mask_pred[:, 1:2, :, :]
        mask_pred, h_next = vismem(appearance, mask_pred, h_next)
        preds.append(mask_pred.data)

        if s == 0:
            loss = loss_calc(rescale(mask_pred), targets[s], args.cuda_vismem)
        else:
            loss = loss + loss_calc(rescale(mask_pred), targets[s], args.cuda_vismem)

    print("{} Iter: {} Loss: {:.4f}".format(datetime.now(), i, loss.data[0]) )
    dot = make_dot(mask_pred)
    dot.render("data/graphs/{}.gv".format(i), view=True)

    if i % display_step == 0:
        pass
        #for p in preds:
        #    plt.imshow(p[0][1].cpu().numpy(), cmap='gray')
        #    plt.show()
    if i % args.save_step == 0:
        torch.save(vismem.state_dict(),'data/models/vismem_'+str(i)+'.pth')

    loss.backward()
    optimizer.step()
