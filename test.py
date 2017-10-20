from networks.deeplab_resnet import Res_Deeplab
from networks.vismem import VisMem
from database import Database
import argparse
import numpy as np
import sys
import torch
import os
import cv2
import math
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc

parser = argparse.ArgumentParser(description='Train the vismem network')
parser.add_argument('--timesteps', metavar='timesteps', type=int, nargs=1, default=3,
                    help='Number of timesteps (frames) to train RNN on')
parser.add_argument('--output_dir', metavar='output_dir', type=str, nargs=1, default="data/DAVIS/Results/Segmentation/480p/out",
                    help='Output dir to save to')
parser.add_argument('--cuda_vismem', metavar='cuda_vismem', type=bool, nargs=1, default=True,
                    help='True if vismem should be run on cuda cores')
parser.add_argument('--cuda_deeplab', metavar='cuda_deeplab', type=bool, nargs=1, default=False,
                    help='True if deeplab should be run on cuda cores')
parser.add_argument('--DAVIS_base', metavar='DAVIS_base', type=str, nargs=1, default="data/DAVIS",
                    help='Location of DAVIS')
parser.add_argument('--test_set', metavar='image_set', type=str, nargs=1, default="data/DAVIS/ImageSets/480p/val.txt",
                    help='Location of the list of training pairs')
args = parser.parse_args()

database = Database(args.DAVIS_base, "data/DAVIS/ImageSets/480p/val.txt")
vismem = VisMem(2048,128,128,7,args.cuda_vismem)
vismem.load_state_dict(torch.load("data/models/vismem_220.pth", map_location=lambda storage, loc: storage))
if args.cuda_vismem: vismem.cuda()

deep_lab = Res_Deeplab()
deep_lab.load_state_dict( torch.load("data/models/MS_DeepLab_resnet_pretrained_COCO_init.pth"))
if args.cuda_deeplab: deep_lab.cuda()
logsoftmax = nn.LogSoftmax()

while database.has_next():
    images, targets, name = database.get_test()
    rescale = nn.UpsamplingBilinear2d(size = ( images[0].shape[2], images[0].shape[3] ))

    preds = []
    for s in range(len(images)-1):
        image = Variable(torch.from_numpy(images[s]).float(), requires_grad=False)
        mask = Variable(torch.from_numpy(targets[s]), requires_grad=False).float()

        if args.cuda_deeplab: image = image.cuda()
        appearance = deep_lab(image)[3]
        appearance = Variable(appearance.data, requires_grad=False)
        if args.cuda_vismem:
            appearance = appearance.cuda()
            mask = mask.cuda()

        if s == 0:
            h_next = None
            mask_pred = mask
        else:
            mask_pred = (math.e**logsoftmax(Variable(mask_pred.data, requires_grad=False)))[:, 1:2, :, :]
            preds.append(rescale(mask_pred).data)
            h_next = Variable(h_next.data, requires_grad=False)
        mask_pred, h_next = vismem(appearance, mask_pred, h_next)

    for idx, p in enumerate(preds):
        p = p[0][0].cpu().numpy()
        m = np.zeros(p.shape)
        m[p>0.6] = 1
        if not os.path.exists(os.path.join(args.output_dir, name, "mask")): os.makedirs(os.path.join(args.output_dir, name, "mask"))
        if not os.path.exists(os.path.join(args.output_dir, name, "probability")): os.makedirs(os.path.join(args.output_dir, name, "probability"))
        scipy.misc.imsave(os.path.join(args.output_dir, name,"mask", "{:05d}.png".format(idx)), m)
        scipy.misc.imsave(os.path.join(args.output_dir, name,"probability", "{:05d}.png".format(idx)), p)
        print(p.shape)
        #plt.imshow(m, cmap='gray')
        #plt.show()
