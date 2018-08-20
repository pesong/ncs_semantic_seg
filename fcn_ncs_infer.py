import os
import time

import numpy
import skimage.io
import skimage.transform
from PIL import Image
import mvnc.mvncapi as mvnc
from utils import vis
import matplotlib.pyplot as plt


# input parameters
IMAGE_MEAN = [71.60167789, 82.09696889, 72.30608881]

GRAPH_PATH = 'ncs_model/mobilenetv2.graph'
IMAGE_PATH_ROOT = 'demo_test/CS/'
IMAGE_DIM = [320, 480]

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
fig = plt.figure(figsize=(18,12))
fig.tight_layout()
plt.subplots_adjust(left=0.04, top= 0.96, right = 0.96, bottom = 0.04, wspace = 0.01, hspace = 0.01)  # 调整子图间距
plt.ion()

i = 0
start = time.time()
for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):

    img_ori = skimage.io.imread(os.path.join(IMAGE_PATH_ROOT + IMAGE_PATH))

    # Resize image [Image size is defined during training]
    img = skimage.transform.resize(img_ori, IMAGE_DIM, preserve_range=True)

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype(numpy.float16)
    image_t = (img - numpy.float16(IMAGE_MEAN))
    image_t = numpy.transpose(image_t, (2, 0, 1))

# -----------step4: get result-------------------------------------------------
    graph.LoadTensor(image_t, 'user object')

    # Get the results from NCS
    out = graph.GetResult()[0]

    #  flatten ---> image
    out = out.reshape(-1, 2).T.reshape(2, 331, -1)
    out = out.argmax(axis=0)
    out = out[6:-5, 6:-5]

    # save result
    voc_palette = vis.make_palette(2)
    out_im = Image.fromarray(vis.color_seg(out, voc_palette))
    iamge_name = IMAGE_PATH.split('/')[-1].rstrip('.jpg')
    # out_im.save('demo_test/' + iamge_name + '_ncs_' + '.png')

    # get masked image
    img_masked = Image.fromarray(vis.vis_seg(img_ori, out, voc_palette))
    # masked_im.save('demo_test/visualization.jpg')

    i += 1
    duration = time.time() - start
    floaps = i / duration
    print("time:{}, images_num:{}, floaps:{}".format(duration, i, floaps))


    # draw picture
    plt.suptitle('MobilenetV2-movidius', fontsize=16)

    plt.subplot(1, 2, 1)
    plt.title("orig image", fontsize=16)
    plt.imshow(img_ori)

    plt.subplot(1, 2, 2)
    plt.title("segmentation", fontsize=16)
    plt.imshow(img_masked)

    plt.pause(0.000001)
    plt.clf()

plt.ioff()
plt.show()
