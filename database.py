import os
import cv2
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Database(object):
    def __init__(self, DAVIS_base, image_set):
        with open(image_set) as f:
            pairs = [l.split() for l in f]
        seq_cur = pairs[0][0].split('/')[-2]
        print(seq_cur)
        sequences = [[]]
        seq_num = 0
        for p in pairs:
            if p[0].split('/')[-2] == seq_cur:
                sequences[seq_num].append(p)
            else:
                seq_cur = p[0].split('/')[-2]
                sequences.append([])
                seq_num = seq_num+1
                sequences[seq_num].append(p)

        self.data_aug_scales = [0.5, 0.5, 1, 1.25, 1.5]
        self.DAVIS_base = DAVIS_base
        self.sequences = sequences
        #random.shuffle(self.sequences)
        self.cur_seq = 0
        #print(self.sequences)

    def has_next(self):
        if self.cur_seq >= len(self.sequences):
            return False
        else:
            return True

    def get_test(self):
        sequence = self.sequences[self.cur_seq]
        self.cur_seq = self.cur_seq + 1
        images = []
        labels = []
        for i in range(0, len(sequence)):
            images.append(self.load_image(os.path.join(self.DAVIS_base, sequence[i][0][1:]), 1, 0))
            labels.append(self.load_mask(os.path.join(self.DAVIS_base, sequence[i][1][1:]), 1, 0))
        name = sequence[0][0].split('/')[-2]
        return images,labels,name

    def get_next_masktrack(self, batch_size):
        sources = []
        targets = []
        scale = self.data_aug_scales[random.randint(0, len(self.data_aug_scales)-1)]
        for sample in range(batch_size):
            images, labels = self.get_next(2, flip_on=True, crop=321, scale=scale)
            sources.append(np.concatenate((images[1], labels[0]), axis = 1))
            targets.append(labels[1])
        sources = np.concatenate(sources, axis = 0)
        targets = np.concatenate(targets, axis = 0)
        return (sources, targets)

    def get_next(self, seq_num, flip_on=False, crop=0, scale=1):

        if flip_on:
             flip = random.randint(0,1)
        else: flip = 0
        seq = random.randint(0, len(self.sequences)-1)
        images = []
        labels = []

        num = len(self.sequences[seq])-1
        subseq = random.randint(0, num-seq_num)
        offset = (random.random(), random.random())
        for i in range(subseq, subseq+seq_num):
            images.append(self.load_image(os.path.join(self.DAVIS_base, self.sequences[seq][i][0][1:]), scale, flip))
            labels.append(self.load_mask(os.path.join(self.DAVIS_base, self.sequences[seq][i][1][1:]), scale, flip))
            if crop:
                shape = images[-1].shape
                coords = ( int(offset[0]*(shape[2]-crop)) if crop<shape[2] else 0,
                           int(offset[0]*(shape[3]-crop)) if crop<shape[3] else  0)
                images[-1] = images[-1][:, :, coords[0]:coords[0]+crop, coords[1]:coords[1]+crop]
                labels[-1] = labels[-1][:, :, coords[0]:coords[0]+crop, coords[1]:coords[1]+crop]

                # plt.imshow(images[-1][0].transpose(1,2,0))
                # plt.show()
                # plt.imshow(labels[-1][0][0]+127.5)
                # plt.show()


        return images, labels


    def load_image(self, imdir, scale, flip):
        #print(imdir)
        img = Image.open(imdir)
        img.load()
        img_size = tuple([int(img.size[0] * scale), int(img.size[1] * scale)])
        #img = img.resize(img_size)
        #if flip == 1: img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.array(img, dtype=np.float32)

        img = np.flip(img, 2).copy()
        #plt.imshow(img/255)
        #plt.show()
        img[:,:,0] = img[:,:,0] - 104.008
        img[:,:,1] = img[:,:,1] - 116.669
        img[:,:,2] = img[:,:,2] - 122.675
        img = img[np.newaxis, :].transpose(0, 3, 1, 2)
        #print(imdir)
        img = cv2.imread(imdir).astype(float)
        print(img[100,100,0])
        print(img[100,100,1])
        print(img[100,100,2])
        print(img.shape)
        #if flip == 1: img = np.flip(img, 1)
        #img[:,:,0] = img[:,:,0] - 104.008
        #img[:,:,1] = img[:,:,1] - 116.669
        #img[:,:,2] = img[:,:,2] - 122.675

        #img = img[np.newaxis, :].transpose(0,3,1,2)

        return img

    def load_mask(self, maskdir, scale, flip):
        img = Image.open(maskdir)
        img.load()
        img_size = tuple([int(img.size[0] * scale), int(img.size[1] * scale)])
        img = img.resize(img_size)
        if flip == 1: img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.split()[0]
        img = np.array(img, dtype=np.int16)
        img = img[np.newaxis, :]
        img = img[np.newaxis, :]
        #img[img == 255] = 1
        img = img-127.5
        #plt.imshow(img[0][0])
        #plt.show()
        return img
