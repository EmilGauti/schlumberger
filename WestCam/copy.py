import glob
import os
import shutil

image_name = []
path="diffImages/labels/project-1-at-2022-11-23-16-02-a86300ac/"
path2="diffImages/images/"
copy2path="images_for_training/"
for mask_image in glob.glob(path+"*png"):
    image_name.append(mask_image.split("/")[-1])

for source_image in glob.glob(path2+"*png"):
    image = source_image.split("/")[-1]
    if image in image_name:
        shutil.copy(source_image,copy2path+image)
