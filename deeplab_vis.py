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


database = Database("data/DAVIS/", "data/DAVIS/ImageSets/480p/val.txt")


deep_lab = Res_Deeplab()
deep_lab.load_state_dict( torch.load("data/models/MS_DeepLab_resnet_pretrained_COCO_init.pth"))
logsoftmax = nn.LogSoftmax()

while database.has_next():
    images, targets, name = database.get_test()
    image = Variable(torch.from_numpy(images[1]).float(), volatile=True)
    rescale = nn.UpsamplingBilinear2d(size = ( images[0].shape[2], images[0].shape[3] ))

    appearance = deep_lab(image)[3]
    appearance = rescale(appearance)
    appearance = appearance.data.numpy()[0]
    print(np.amin(appearance))
    print(np.amax(appearance))
    print(appearance.shape)
    slices = np.split(appearance, appearance.shape[0], axis=0)
    print(slices[0].shape)
    overlay_color = [255, 0, 0]
    transparency = 0.999
    maxval = np.amax(appearance)
    slices = sorted(slices, key=lambda x: np.amax(x))
    os.makedirs(os.path.join("deeplabvis", name))
    for i in range(len(slices)):
        #rescaled = rescale(Variable(torch.from_numpy(slices[i][0]))).data.numpy()
        masked = images[1][0] + slices[i][0]
        img = images[1][0].transpose(1,2,0)
        mask = (slices[i][0] / maxval)
        im_over = np.ndarray(img.shape)
        im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
        im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
        im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])
        print(masked.shape)
        scipy.misc.imsave(os.path.join("deeplabvis", name, str(i) + ".jpg"), im_over)

'''
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

'''
