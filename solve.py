
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

weights = 'weight_pretrained/mobilenet_v2_ncs.caffemodel'
proto = 'weight_pretrained/mobilenet_v2_ncs_deploy.prototxt'
final_model_name = 'cityscapes_4s_mobilenet_v2'
n_steps = 20000

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
mobilenetv2_net = caffe.Net(proto, weights, caffe.TRAIN)
surgery.transplant(solver.net, mobilenetv2_net)
del mobilenetv2_net

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/dl/data/cityscapes/cityscapes_ncs/val_test.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
