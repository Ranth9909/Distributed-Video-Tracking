import os
import uuid
from sklearn.model_selection import train_test_split
from glob import glob
import shutil
from PIL import Image
image_files = glob("D:/ULEARN/ULEARN(1)/4TH 2ND/FYP_1/frames/ID0_/*.png")
images = [name.replace(".png","") for name in image_files]
train_names, test_names = train_test_split(images, test_size=0.7)

def batch_move_files(file_list, source_path, destination_path):
    for file in file_list:
        image = file.split('\\')[1] + '.png'
        shutil.copy(os.path.join(source_path, image), destination_path)
    return

source_dir = "PATH/frames/ID0_/"
pos = "PATH/positive"
anc = "PATH/anchor"
neg_source = "PATH/PKUv1a_128x48/"

batch_move_files(train_names, source_dir, pos)
batch_move_files(test_names, source_dir, anc)

path = 'PATH/positive'
files = os.listdir(path)
for index, file in enumerate(files):
   os.rename(os.path.join(path, file), os.path.join(path, '{}.png'.format(uuid.uuid1())))

path1 = "PATH/negative/"
dirs = os.listdir(path1)

def resize():
    for item in dirs:
        if os.path.isfile(path1+item):
            im = Image.open(path1+item)
            f, e = os.path.splitext(path1+item)
            imResize = im.resize((128,128), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)

resize()