import os

base_path = '/dl/data/camVid/camVid_ncs/'

in_f = open(base_path + 'test.txt', 'r')
out_f = open(base_path + 'test1.txt', 'w')

for line in in_f.readlines():
    train_file_name = line.split(" ")[0].lstrip("/SegNet/CamVid/test/")
    out_f.write(train_file_name + '\n')

