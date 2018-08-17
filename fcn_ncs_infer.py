import os

import numpy
import skimage.io
import skimage.transform
from PIL import Image
import mvnc.mvncapi as mvnc
from utils import vis
import matplotlib.pyplot as plt


# input parameters
IMAGE_MEAN = [71.60167789, 82.09696889, 72.30608881]

GRAPH_PATH = 'ncs_model/Inception_fcn4s_city.graph'
IMAGE_PATH_ROOT = 'demo_test/CS/'
IMAGE_DIM = [320, 480]

# -----------------open the device and get a handle to it--------------------
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



for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):

    img_draw = skimage.io.imread(os.path.join(IMAGE_PATH_ROOT + IMAGE_PATH))

    # Resize image [Image size is defined during training]
    img = skimage.transform.resize(img_draw, IMAGE_DIM, preserve_range=True)

    # Convert RGB to BGR [skimage reads image in RGB, some networks may need BGR]
    img = img[:, :, ::-1]

    # Mean subtraction & scaling [A common technique used to center the data]
    img = img.astype(numpy.float16)
    image_t = (img - numpy.float16(IMAGE_MEAN))

    # -----------step4: get result-------------------------------------------------
    graph.LoadTensor(image_t, 'user object')

    # Get the results from NCS
    out = graph.GetResult()[0]

    #  flatten ---> image
    out = out.reshape(-1, 2).T.reshape(2, 331, -1)
    out = out.argmax(axis=0)
    out = out[:-11, :-11]
    # print(out)



    # save result
    voc_palette = vis.make_palette(2)
    out_im = Image.fromarray(vis.color_seg(out, voc_palette))
    iamge_name = IMAGE_PATH.split('/')[-1].rstrip('.jpg')
    out_im.save('demo_test/' + iamge_name + '_ncs_' + '.png')

    masked_im = Image.fromarray(vis.vis_seg(img_draw, out, voc_palette))
    plt.imshow(masked_im)
    plt.show()


    # masked_im.save('demo_test/visualization.jpg')


