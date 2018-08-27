import os
import time
import cv2
import numpy
import skimage.io
import vis
import skimage.transform
from PIL import Image
import mvnc.mvncapi as mvnc
import matplotlib.pyplot as plt

# input parameters
IMAGE_MEAN = [71.60167789, 82.09696889, 72.30608881]
GRAPH_PATH = '/dl/model/seg/caffe/ncs_fcns/ncs_model/Inception_fcn4s_road_0827.graph'
IMAGE_PATH_ROOT = '/dl/model/seg/caffe/ncs_fcns/demo_test/CS/'
IMAGE_DIM = [320, 480]

Video = cv2.VideoCapture(0)

# --------step1: open the device and get a handle to it--------------------
# look for device
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print("No devices found")
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

# ---------step2: load a graph file into hte ncs device----------------------
# read the graph file into a buffer
with open(GRAPH_PATH, mode='rb') as f:
    blob = f.read()

# Load the graph buffer into the ncs
graph = device.AllocateGraph(blob)

# -------- step3: offload image into the ncs to run inference
# fig = plt.figure(figsize=(20,10))
# plt.subplots_adjust(left=0.04, top= 0.96, right = 0.96, bottom = 0.04, wspace = 0.01, hspace = 0.01)
# plt.ion()

i = 0
start = time.time()


while True:

    ret, img_ori = Video.read()

    # img = img[:, :, ::-1]
    # img = skimage.transform.resize(img, IMAGE_DIM)
    # img = img * 255
    # # img = img.astype(numpy.float16)
    # # image_t = (img - numpy.float16(IMAGE_MEAN))
    #
    # cv2.imshow("talker", img)
    # cv2.waitKey(3)
#
#
# for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):
#
#     img_ori = skimage.io.imread(os.path.join(IMAGE_PATH_ROOT + IMAGE_PATH))

    # Resize image [Image size is defined during training]
    img_resize = skimage.transform.resize(img_ori, IMAGE_DIM) * 255

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    image_t = img_resize[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    image_t = image_t.astype(numpy.float16)
    image_t = (image_t - numpy.float16(IMAGE_MEAN))

# -----------step4: get result-------------------------------------------------
    graph.LoadTensor(image_t, 'user object')

    # Get the results from NCS
    out = graph.GetResult()[0]

    #  flatten ---> image
    out = out.reshape(-1, 2).T.reshape(2, 331, -1)
    out = out.argmax(axis=0)
    out = out[:-11, :-11]

    # save result
    voc_palette = vis.make_palette(2)
    out_im = Image.fromarray(vis.color_seg(out, voc_palette))
    # iamge_name = IMAGE_PATH.split('/')[-1].rstrip('.jpg')
    # out_im.save('demo_test/' + iamge_name + '_ncs_' + '.png')

    # get masked image
    img_masked = vis.vis_seg(img_resize, out, voc_palette)
    # masked_im.save('demo_test/visualization.jpg')

    i += 1
    duration = time.time() - start
    floaps = i / duration
    print("time:{}, images_num:{}, floaps:{}".format(duration, i, floaps))


    # # visualization
    # plt.suptitle('inception-movidius', fontsize=16)
    #
    # plt.subplot(1, 2, 1)
    # plt.title("original image", fontsize=16)
    cv2.imshow("ori", img_ori)

    cv2.waitKey(3)
    # plt.subplot(1, 2, 2)
    # plt.title("segmentation", fontsize=16)
    cv2.imshow("seg", img_masked)

    # plt.pause(0.01)
    # plt.clf()

# plt.ioff()
# plt.show()









