import os
from PIL import Image
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import cv2

gs_iamge_path = '/dl/data/test/'

for file in os.listdir(gs_iamge_path):
    file_name = str(file).rstrip('.jpeg')
    file_path = gs_iamge_path + str(file)
    img = cv2.imread(file_path)
    # b, g, r = cv2.split(img)
    # out_img = cv2.merge([r, g, b])
    out_img = cv2.resize(img, (480, 320), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("/dl/data/test/" + str(file_name) + '.jpg', out_img)

