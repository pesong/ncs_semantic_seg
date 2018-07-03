import os

base_path = '/dl/data/kitti_road/data_road_ncs/'

in_f = open(base_path + 'val3.txt', 'r')
out_f = open(base_path + 'val.txt', 'w')

for line in in_f.readlines():
    train_file_name = line.split(" ")[0].lstrip("training/image_2/")
    out_f.write(train_file_name + '\n')

