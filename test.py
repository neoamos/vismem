from networks.deeplab_resnet import Res_Deeplab
from networks.vismem import VisMem
from networks.deeplab_masktrack import Deeplab_Masktrack
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='Train the vismem network')
parser.add_argument('--timesteps', metavar='timesteps', type=int, nargs=1, default=3,
                    help='Number of timesteps (frames) to train RNN on')
parser.add_argument('--output_dir', metavar='output_dir', type=str, nargs=1, default="data/DAVIS/Results/Segmentation/480p/masktrack_v22",
                    help='Output dir to save to')
parser.add_argument('--cuda_vismem', metavar='cuda_vismem', type=bool, nargs=1, default=True,
                    help='True if vismem should be run on cuda cores')
parser.add_argument('--cuda_deeplab', metavar='cuda_deeplab', type=bool, nargs=1, default=True,
                    help='True if deeplab should be run on cuda cores')
parser.add_argument('--DAVIS_base', metavar='DAVIS_base', type=str, nargs=1, default="data/DAVIS",
                    help='Location of DAVIS')
parser.add_argument('--test_set', metavar='image_set', type=str, nargs=1, default="data/DAVIS/ImageSets/480p/val.txt",
                    help='Location of the list of training pairs')
args = parser.parse_args()

database = Database(args.DAVIS_base, "data/DAVIS/ImageSets/480p/val.txt")
#vismem = VisMem(2048,2049,1024,7,args.cuda_vismem)
#vismem.load_state_dict(torch.load("data/models/bigmem2/vismem_29900.pth", map_location=lambda storage, loc: storage))
#if args.cuda_vismem: vismem.cuda()

#deep_lab = Res_Deeplab()
#deep_lab.load_state_dict( torch.load("data/models/bigmem2/deep_lab_29900.pth"))
#deep_lab.load_state_dict( torch.load("data/models/MS_DeepLab_resnet_pretrained_COCO_init.pth"))
#if args.cuda_deeplab: deep_lab.cuda()
logsoftmax = nn.LogSoftmax()

deep_lab = Deeplab_Masktrack()
deep_lab.load_state_dict(torch.load("data/models/masktrack_v22/masktrack_30000.pth"))
if args.cuda_deeplab: deep_lab.cuda()


def vismem_pass(images, target):
    print(len(images))
    rescale = nn.UpsamplingBilinear2d(size = ( images[0].shape[2], images[0].shape[3] ))
    print(name)
    preds = []
    for s in range(len(images)):
        image = Variable(torch.from_numpy(images[s]).float(), volatile=True)
        mask = Variable(torch.from_numpy(targets[s]).float(), volatile=True)

        if args.cuda_deeplab: image = image.cuda()
        appearance = deep_lab(image)[3]
        appearance = Variable(appearance.data, requires_grad=False)
        if args.cuda_vismem:
            appearance = appearance.cuda()
            mask = mask.cuda()

        if s == 0:
            h_next = None
            mask_pred = Variable(torch.zeros(mask.data.size())).cuda()
        else:
            mask_pred = (math.e**logsoftmax(Variable(mask_pred.data, requires_grad=False)))[:, 1:2, :, :]
            #mask_pred = mask
            h_next = Variable(h_next.data, requires_grad=False)
        mask_pred, h_next = vismem(appearance, mask_pred, h_next)
        preds.append(rescale(mask_pred).data)

def masktrack_pass(images, target):
    rescale = nn.UpsamplingBilinear2d(size = ( images[0].shape[2], images[0].shape[3] ))
    target = torch.from_numpy(target).float().cuda()
    preds = [target]
    for i in range(1, len(images)):
        image = torch.from_numpy(images[i]).float().cuda()
#        print(image.size())
#        print(type(image))
#        print(type(preds[-1]))
#        print(preds[-1].size())
        source = Variable( torch.cat((image,preds[-1]), dim=1) , volatile = True ).cuda()
        mask_pred = deep_lab(source)[0].data[0][None, :, :, :]
        mask_pred = (math.e**logsoftmax(Variable(mask_pred, requires_grad=False)))[:, 1:2, :, :]
        mask_pred = 255*(mask_pred-0.5)
        preds.append(rescale(mask_pred).data[0][None, :, :, :])
    return preds

while database.has_next():
    images, targets, name = database.get_test()
    print(name)
    #preds = vismem_pass(images, targets[0])
    preds = masktrack_pass(images, targets[0])

    for idx, p in enumerate(preds):
        p = p[0][0].cpu().numpy()
        m = np.zeros(p.shape)
        m = p.astype(np.float32)>(162.0/255.0)
        m = m.astype(np.float32)
        if not os.path.exists(os.path.join(args.output_dir, "mask", name)): os.makedirs(os.path.join(args.output_dir, "mask",  name))
        if not os.path.exists(os.path.join(args.output_dir, "probability",name)): os.makedirs(os.path.join(args.output_dir, "probability", name))
        scipy.misc.imsave(os.path.join(args.output_dir,"mask", name, "{:05d}.png".format(idx)), m)
        scipy.misc.imsave(os.path.join(args.output_dir,"probability", name, "{:05d}.png".format(idx)), p)
        #plt.imshow(m, cmap='gray')
        #plt.show()
