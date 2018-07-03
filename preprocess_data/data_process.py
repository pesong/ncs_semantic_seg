import os
from PIL import Image

# path_name = '/dl/data/bdd100k/seg/images/test'
# train_list_file = open('/dl/data/bdd100k/seg/images/test.txt', 'w')
#
# for file in os.listdir(path_name):
#     file_name = str(file).rstrip('.jpg')
#     train_list_file.write(file_name + '\n')

path_name = '/media/pesong/e/dl_gaussian/data/bdd100k/seg/images/total_images/'

for file in os.listdir(path_name):
    file_path = path_name + str(file)
    with Image.open(file_path) as img:
        resized = img.resize((480, 320))
        resized.save("/media/pesong/e/dl_gaussian/data/bdd100k/seg/images/total_320_480/" + str(file))