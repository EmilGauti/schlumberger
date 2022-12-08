import glob
import os
path="diffImages/labels/project-1-at-2022-11-23-16-02-a86300ac/"
for target_image in glob.glob(path+"*png"):
    print(target_image)
    image_nr = target_image.split("/")[-1]
    image_nr = image_nr.split(".")[0]
    os.rename(target_image,path+image_nr+"_target.png")