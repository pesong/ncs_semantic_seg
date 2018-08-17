import os
import cv2
import sys
sys.path.append('/opt/movidius/caffe/python')
import caffe
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random



class CityScapeSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - bdd_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for PASCAL VOC semantic segmentation.
        example
        params = dict(bdd_dir="/path/to/bdd100k/seg",
            mean=[71.60167789, 82.09696889, 72.30608881],
            split="val")
        """
        # config
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        self.cityscape_dir = params['cityscape_dir']
        self.batch_size = params['batch_size']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.cropsize = params['crop_size']

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f = '{}/{}.txt'.format(self.cityscape_dir, self.split)

        self.folders = open(split_f, 'r').read().splitlines()

        self.imglist = []
        self.labellist = []

        path_imgs = self.folders[0]
        for root, dirs, files in os.walk(path_imgs):
            for dir in dirs:
                for name in os.listdir(root + dir):
                    if ('leftImg8bit' in name):
                        name_label = name.replace('leftImg8bit', 'gtFine_labelIds')
                        root_label = root.replace(self.folders[0], self.folders[1])
                        self.imglist.append(root + dir + '/' + name)
                        self.labellist.append(root_label + dir + '/' + name_label)

        # get list of image indexes.
        self.idx = 0  # current image

        # make eval deterministic
        if 'train' not in params['split']:
            self.random = False

        # randomization: seed
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.imglist) - 1)

    def reshape(self, bottom, top):

        top[0].reshape(self.batch_size, 3, self.cropsize[0], self.cropsize[1])
        top[1].reshape(self.batch_size, 1, self.cropsize[0], self.cropsize[1])

    def forward(self, bottom, top):

        for itt in range(self.batch_size):

            #  randomization: seed and pick
            if self.random:
                self.idx = random.randint(0, len(self.imglist) - 1)
            else:
                self.idx += 1
                if self.idx == (len(self.imglist) - 1):
                    self.idx = 0

            # use the batch loader to load the next image
            # print(self.idx)
            self.data = self.load_image()
            self.label = self.load_label()

            top[0].data[itt, ...] = self.data
            top[1].data[itt, ...] = self.label

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.imglist[self.idx])
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_label(self):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """

        # cityscapes  读取灰度图，对多分类进行相应的映射
        label = Image.open(self.labellist[self.idx])
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]

        label_road = np.all(label == [7], axis=0)
        label_bg = np.any(label != [7], axis=0)

        label_all = np.dstack([label_bg, label_road])
        label_all = label_all.astype(np.float32)
        label_all = label_all.transpose((2, 0, 1))
        label_all = label_all[0]

        # plt.imshow(label_all)
        # plt.show()

        label = label_all[np.newaxis, ...]
        return label




class KittiSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - bdd_dir: path to PASCAL VOC year dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for PASCAL VOC semantic segmentation.
        example
        params = dict(bdd_dir="/path/to/bdd100k/seg",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.kitti_dir = params['kitti_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.kitti_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/train_320_480/{}'.format(self.kitti_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        idx = idx.split('_')
        idx.insert(1, '_road_')
        idx = ''.join(idx)
        idx_file = '{}/label_320_480/{}'.format(self.kitti_dir, idx)

        label = Image.open(idx_file)
        label = np.array(label, dtype=np.uint8)
        label = label[:, :, ::-1]

        label_road = np.all(label == [255, 0, 255], axis=2)
        label_bg = np.any(label != [255, 0, 255], axis=2)

        label_all = np.dstack([label_bg, label_road])
        label_all = label_all.astype(np.float32)
        label_all = label_all.transpose((2, 0, 1))
        label_all = label_all[0]

        # plt.imshow(label_all)
        # plt.show()

        label = label_all[np.newaxis, ...]

        return label


class SBDDSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the SBDD extended labeling
    of PASCAL VOC for semantic segmentation
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - sbdd_dir: path to SBDD `dataset` dir
        - split: train / seg11valid
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        for SBDD semantic segmentation.
        N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
        Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.
        example
        params = dict(sbdd_dir="/path/to/SBDD/dataset",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="valid")
        """
        # config
        params = eval(self.param_str)
        self.sbdd_dir = params['sbdd_dir']
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.sbdd_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        import scipy.io
        mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        label = label[np.newaxis, ...]
        return label