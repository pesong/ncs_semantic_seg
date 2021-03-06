import sys
sys.path.insert(0, '/opt/movidius/caffe/python')
import caffe

import numpy
from PIL import Image
import matplotlib.pyplot as plt
from utils import vis

# define parameters
IMAGE_PATH = 'demo_test/gaussian/2.jpg'

IMAGE_MEAN = [104.00698793,116.66876762,122.67891434]
IMAGE_DIM = [320, 480]

NET_PROTO = 'deploy.prototxt'
# WEIGHTS = 'fcn-alexnet-pascal.caffemodel'
WEIGHTS = 'snapshot/googlenet_8s_kitti/solver_iter_100000.caffemodel'
# WEIGHTS = 'weight_pretrained/bvlc_googlenet.caffemodel'


# load net
net = caffe.Net(NET_PROTO, WEIGHTS, caffe.TEST)

# ------------------1: handle image use numpy

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
# img = Image.open('demo_test/bicycle.jpg')
# img = np.array(im, dtype=np.float32)
# img = in_[:,:,::-1]
# img -= np.array((104.00698793,116.66876762,122.67891434))
# image_t = in_.transpose((2,0,1))
#
# # shape for input (data blob is N x C x H x W), set data
# net.blobs['data'].data[...] = image_t

# ------------------1: handle image use skimage(by ncs)

# img_draw = skimage.io.imread(IMAGE_PATH)
#
# # Resize image [Image size is defined during training]
# img = skimage.transform.resize(img_draw, IMAGE_DIM, preserve_range=True)
#
# # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
# img = img[:, :, ::-1]
#
# # Mean subtraction & scaling [A common technique used to center the data]
# img = img.astype(numpy.float16)
# image_t = (img - numpy.float16(IMAGE_MEAN))
# image_t = image_t.transpose((2,0,1))



# # ---------------3: handle image use caffe
caffe_root = '/opt/movidius/caffe/'

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = numpy.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': (1,3,320,480)})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

img = Image.open(IMAGE_PATH)
image = caffe.io.load_image(IMAGE_PATH)
image_t = transformer.preprocess('data', image)

# # copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = image_t

# ------------------infer-----------------------------
# run net and take argmax for prediction
net.forward()
out = net.blobs['upscore'].data[0]
out = out.argmax(axis=0)
out = out[:-11, :-11]

plt.imshow(out)
plt.show()

# visualize segmentation in PASCAL VOC colors
voc_palette = vis.make_palette(2)
out_im = Image.fromarray(vis.color_seg(out, voc_palette))
iamge_name = IMAGE_PATH.split('/')[-1].rstrip('.jpg')
out_im.save('demo_test/' + iamge_name + '_pc_' + '.png')

masked_im = Image.fromarray(vis.vis_seg(img, out, voc_palette))
masked_im.save('demo_test/visualization.jpg')