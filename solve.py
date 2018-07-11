
import sys
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

weights = 'weight_pretrained/bvlc_googlenet.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/dl/data/kitti_road/data_road_ncs/val.txt', dtype=str)

for _ in range(25):
    solver.step(1)
    score.seg_tests(solver, False, val, layer='score')
