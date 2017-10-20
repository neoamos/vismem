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

        '''
        seq_names = sorted(os.listdir(image_dir))
        images = []
        annotations = []
        for seq_name in seq_names:
            frames = sorted(os.listdir(os.path.join(image_dir, seq_name)))
            frames = map(lambda x: os.path.join(image_dir, seq_name, x), frames)
            images.append(frames)
            masks = sorted(os.listdir(os.path.join(annotation_dir, seq_name)))
            masks = map(lambda x: os.path.join(annotation_dir, seq_name, x), masks)
            annotations.append(masks)
        '''

        self.data_aug_scales = [0.5, 0.8, 1]
        self.DAVIS_base = DAVIS_base
        self.sequences = sequences
        random.shuffle(self.sequences)
        self.cur_seq = 0

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
        for i in range(0, len(sequence)-1):
            images.append(self.load_image(os.path.join(self.DAVIS_base, sequence[i][0][1:]), 1, 0))
            labels.append(self.load_mask(os.path.join(self.DAVIS_base, sequence[i][1][1:]), 1, 0))
        name = sequence[0][0].split('/')[-2]
        return images,labels,name

    def get_next(self, seq_num):

        scale = self.data_aug_scales[random.randint(0, len(self.data_aug_scales)-1)]
        flip = random.randint(0,1)
        seq = random.randint(0, len(self.sequences)-1)
        images = []
        labels = []

        num = len(self.sequences[seq])-1
        subseq = random.randint(0, num-seq_num)
        for i in range(subseq, subseq+seq_num):
            images.append(self.load_image(os.path.join(self.DAVIS_base, self.sequences[seq][i][0][1:]), scale, flip))
            labels.append(self.load_mask(os.path.join(self.DAVIS_base, self.sequences[seq][i][1][1:]), scale, flip))
        return images, labels


    def load_image(self, imdir, scale, flip):
        #print(imdir)
        #img = Image.open(imdir)
        #img.load()
        #img_size = tuple([int(img.size[0] * scale), int(img.size[1] * scale)])
        #img = img.resize(img_size)
        #if flip == 1: img = img.transpose(Image.FLIP_LEFT_RIGHT)

        #img = np.array(img, dtype=np.float32)
        #img[:,:,0] = img[:,:,0] - 104.008
        #img[:,:,1] = img[:,:,1] - 116.669
        #img[:,:,2] = img[:,:,2] - 122.675
        #img = img[np.newaxis, :].transpose(0, 3, 1, 2)
        #plt.imshow(img[0][0], cmap='gray')
        #plt.show()
        print(imdir)
        img = cv2.imread(imdir).astype(float)
        img[:,:,0] = img[:,:,0] - 104.008
        img[:,:,1] = img[:,:,1] - 116.669
        img[:,:,2] = img[:,:,2] - 122.675
        img = img[np.newaxis, :].transpose(0,3,1,2)
        return img

    def load_mask(self, maskdir, scale, flip):
        img = Image.open(maskdir)
        img.load()
        img_size = tuple([int(img.size[0] * scale), int(img.size[1] * scale)])
        #img = img.resize(img_size)
        #if flip == 1: img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = img.split()[0]
        img = np.array(img, dtype=np.uint8)
        img = img[np.newaxis, :]
        img = img[np.newaxis, :]
        img[img == 255] = 1
        #plt.imshow(img[0][0], cmap='gray')
        #plt.show()
        return img
