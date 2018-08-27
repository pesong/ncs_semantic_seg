
import sys
import matplotlib.pyplot as plt

import numpy

sys.path.append('/home/pesong/tools/caffe/python')

import caffe
from utils import score, surgery

import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

# init train from scratch
# weights = 'snapshot/googlenet_8s_cityscapes/solver_iter_12000.caffemodel'
# proto = 'weight_pretrained/bvlc_googlenet_deploy.prototxt'

# train from fine tune
weights = 'weight_pretrained/inception_fcn4s_city_batch20.caffemodel'
proto = 'weight_pretrained/inception_fcn4s_city_batch20_deploy.prototxt'

final_model_name = 'cityscapes_4s_inception'
n_steps = 20000

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
inception_net = caffe.Net(proto, weights, caffe.TRAIN)
surgery.transplant(solver.net, inception_net)
del inception_net

# solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/dl/data/cityscapes/cityscapes_ncs/val_test.txt', dtype=str)


for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
