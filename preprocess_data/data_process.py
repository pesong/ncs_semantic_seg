import os
from PIL import Image

path_name = '/media/pesong/e/dl_gaussian/data/cityscapes/cityscapes_ori/leftImg8bit/val'
target_name = '/media/pesong/e/dl_gaussian/data/cityscapes/cityscapes_ncs/leftImg8bit/val'
# target_file = '/media/pesong/e/dl_gaussian/data/cityscapes/cityscapes_ncs/val.txt'
# f = open(target_file, 'w')

for root, dirs, files in os.walk(path_name):
    for dir in dirs:

        save_path = target_name + '/' + dir + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        dir_path = root + '/' + dir
        for image_name in os.listdir(dir_path):
            img_path = dir_path + '/' + str(image_name)
            # f.write(img_path + '\n')
            with Image.open(img_path) as img:
                resized = img.resize((480, 320))
                resized.save(save_path + str(image_name))

# f.close()
