
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

weights = 'fcn-alexnet-pascal.caffemodel'
solver = 'solver.prototxt'

# init
caffe.set_device(0)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

solver = caffe.SGDSolver(solver)
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/dl/data/bdd100k/seg/val.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
