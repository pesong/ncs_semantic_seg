
import sys
sys.path.append('/home/pesong/tools/ssd-caffe/python')

import caffe
from utils import score, surgery
import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass


weights = 'snapshot/mobilenetv2_fcn4s_road/solver_iter_40000.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# scoring
val = np.loadtxt('/dl/data/cityscapes/cityscapes_ncs/val_test.txt', dtype=str)

score.seg_tests(solver, False, val, layer='score')


