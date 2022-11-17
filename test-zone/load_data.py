import numpy as np
from PIL import Image
import glob
from torchvision import transforms     
import torch

def load_images():
    test_images = []
    test_labels = []
    
    X_test = torch.zeros([40,231,231], dtype=torch.float32)
    Y_test = torch.zeros([40,231,231], dtype=torch.float32)
    X_train = torch.zeros([80,231,231], dtype=torch.float32)
    Y_train = torch.zeros([80,231,231], dtype=torch.float32)
    convert_tensor = transforms.PILToTensor()
    for i, label_path in enumerate(glob.glob("simulated_data/Test/*_target.png")):
        filename = label_path.split("/")[-1]
        path = label_path.split(filename)[0]
        id = filename.split("_")[0]
        image_path=path+id+".png"

        image = Image.open(image_path).convert('L')
        X_test[i,:,:] = convert_tensor(image)
        
        label = Image.open(label_path).convert('L')
        Y_test[i,:,:] = convert_tensor(label)
    for i, label_path in enumerate(glob.glob("simulated_data/Train/*_target.png")):
        filename = label_path.split("/")[-1]
        path = label_path.split(filename)[0]
        id = filename.split("_")[0]
        image_path=path+id+".png"

        image = Image.open(image_path).convert('L')
        X_train[i,:,:] = convert_tensor(image)
        
        label = Image.open(label_path).convert('L')
        Y_train[i,:,:] = convert_tensor(label)
    
    return X_train, Y_train, X_test, Y_test
#load_images()