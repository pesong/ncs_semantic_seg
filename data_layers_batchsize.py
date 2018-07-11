import cv2
import sys
sys.path.append('/media/ziwei/Harddisk02/HanBin/caffe/python')
import caffe

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import random

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
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # self.kitti_dir = params['kitti_dir']
        # self.split = params['split']
        # self.mean = np.array(params['mean'])
        # self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, 3, params['crop_size'][0], params['crop_size'][1])

        top[1].reshape(self.batch_size, 1, params['crop_size'][0], params['crop_size'][1]) #each image label is a intger

        print_info("ImageDataLayer", params)

        # load indices for images and labels
        # split_f  = '{}/{}.txt'.format(self.kitti_dir, self.split)
        # self.indices = open(split_f, 'r').read().splitlines()
        # self.idx = 0

        # make eval deterministic
        if 'train' not in params['split']:
            self.random = False
        #
        # # randomization: seed and pick
        # if self.random:
        #     random.seed(params['seed'])
        #     self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        # self.data = self.load_image(self.indices[self.idx])
        # self.label = self.load_label(self.indices[self.idx])
        # # reshape tops to fit (leading 1 is for batch dimension)
        # top[0].reshape(1, *self.data.shape)
        # top[1].reshape(1, *self.label.shape)
        pass


    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im = self.batch_loader.load_image()
            label = self.batch_loader.load_label()
            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label


    def backward(self, top, propagate_down, bottom):
        pass


class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.params = params
        self.source = params['source']
        self.batch_size = params['batch_size']
        self.random = params.get('randomize', True)
        self.crop_size = params['crop_size']
        self.split = params['split']

        # get list of image indexes.
        self.split_file = '{}/{}.txt'.format(self.source, self.split)
        self.imagelist = open(self.split_file, 'r').read().splitlines()

        self._cur = 0  # current image
        # this class does some simple data-manipulations
        # self.transformer = SimpleTransformer()

        print("BatchLoader initialized with {} images".format(
            len(self.imagelist)))

        # make eval deterministic
        if 'train' not in params['split']:
            self.random = False
        #
        # # randomization: seed and pick
        if self.random:
            random.seed(params['seed'])
            self.idx = random.randint(0, len(self.imagelist)-1)

    def load_image(self):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/train_320_480/{}'.format(self.source, self.imagelist[self.idx]))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.params['mean']
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        idx = self.imagelist[self.idx].split('_')
        idx.insert(1, '_road_')
        idx = ''.join(idx)
        idx_file = '{}/label_320_480/{}'.format(self.source, idx)

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

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    required = ['batch_size', 'source', 'crop_size']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print("{} initialized for with batch_size: {}, crop_size: {}.".format(
        name,params['batch_size'],params['crop_size']))
    
# class SBDDSegDataLayer(caffe.Layer):
#     """
#     Load (input image, label image) pairs from the SBDD extended labeling
#     of PASCAL VOC for semantic segmentation
#     one-at-a-time while reshaping the net to preserve dimensions.
#     Use this to feed data to a fully convolutional network.
#     """
#
#     def setup(self, bottom, top):
#         """
#         Setup data layer according to parameters:
#         - sbdd_dir: path to SBDD `dataset` dir
#         - split: train / seg11valid
#         - mean: tuple of mean values to subtract
#         - randomize: load in random order (default: True)
#         - seed: seed for randomization (default: None / current time)
#         for SBDD semantic segmentation.
#         N.B.segv11alid is the set of segval11 that does not intersect with SBDD.
#         Find it here: https://gist.github.com/shelhamer/edb330760338892d511e.
#         example
#         params = dict(sbdd_dir="/path/to/SBDD/dataset",
#             mean=(104.00698793, 116.66876762, 122.67891434),
#             split="valid")
#         """
#         # config
#         params = eval(self.param_str)
#         self.sbdd_dir = params['sbdd_dir']
#         self.split = params['split']
#         self.mean = np.array(params['mean'])
#         self.random = params.get('randomize', True)
#         self.seed = params.get('seed', None)
#
#         # two tops: data and label
#         if len(top) != 2:
#             raise Exception("Need to define two tops: data and label.")
#         # data layers have no bottoms
#         if len(bottom) != 0:
#             raise Exception("Do not define a bottom.")
#
#         # load indices for images and labels
#         split_f  = '{}/{}.txt'.format(self.sbdd_dir,
#                 self.split)
#         self.indices = open(split_f, 'r').read().splitlines()
#         self.idx = 0
#
#         # make eval deterministic
#         if 'train' not in self.split:
#             self.random = False
#
#         # randomization: seed and pick
#         if self.random:
#             random.seed(self.seed)
#             self.idx = random.randint(0, len(self.indices)-1)
#
#
#     def reshape(self, bottom, top):
#         # load image + label image pair
#         self.data = self.load_image(self.indices[self.idx])
#         self.label = self.load_label(self.indices[self.idx])
#         # reshape tops to fit (leading 1 is for batch dimension)
#         top[0].reshape(1, *self.data.shape)
#         top[1].reshape(1, *self.label.shape)
#
#
#     def forward(self, bottom, top):
#         # assign output
#         top[0].data[...] = self.data
#         top[1].data[...] = self.label
#
#         # pick next input
#         if self.random:
#             self.idx = random.randint(0, len(self.indices)-1)
#         else:
#             self.idx += 1
#             if self.idx == len(self.indices):
#                 self.idx = 0
#
#
#     def backward(self, top, propagate_down, bottom):
#         pass
#
#
#     def load_image(self, idx):
#         """
#         Load input image and preprocess for Caffe:
#         - cast to float
#         - switch channels RGB -> BGR
#         - subtract mean
#         - transpose to channel x height x width order
#         """
#         im = Image.open('{}/img/{}.jpg'.format(self.sbdd_dir, idx))
#         in_ = np.array(im, dtype=np.float32)
#         in_ = in_[:,:,::-1]
#         in_ -= self.mean
#         in_ = in_.transpose((2,0,1))
#         return in_
#
#
#     def load_label(self, idx):
#         """
#         Load label image as 1 x height x width integer array of label indices.
#         The leading singleton dimension is required by the loss.
#         """
#         import scipy.io
#         mat = scipy.io.loadmat('{}/cls/{}.mat'.format(self.sbdd_dir, idx))
#         label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
#         label = label[np.newaxis, ...]
#         return label

