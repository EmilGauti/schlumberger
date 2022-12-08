import glob
import json
from PIL import Image
import os
task2image = {}
for json_file in glob.glob("diffImages/labels/*.json"):
    f=open(json_file)
    j = json.load(f)
    task2image[j["task"]["id"]] = j["task"]["data"]["image"].split("/")[-1]
#print(task2image[972])
#path="diffImages/images/"
#image = Image.open(path+task2image[575])
#image.show()
#for key in task2image.keys():
#    print(key)
path="diffImages/labels/project-1-at-2022-11-23-16-02-a86300ac/"
for mask_image in glob.glob(path+"*png"):
    id = mask_image.split("task-")[-1]
    id = int(id.split("-annotation")[0])
    os.rename(mask_image,path+task2image[id])
    print(id, task2image[id])

