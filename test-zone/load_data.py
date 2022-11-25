import numpy as np
from PIL import Image
import glob
from torchvision import transforms
import torchvision     
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import random

def load_images():
    test_images = []
    test_labels = []
    
    #X_test = torch.zeros([0,231,231], dtype=torch.float32)
    #Y_test = torch.zeros([0,231,231], dtype=torch.float32)
    X_train = torch.zeros([152,210,381], dtype=torch.float32)
    Y_train = torch.zeros([152,210,381], dtype=torch.float32)
    convert_tensor = transforms.PILToTensor()
    """
    for i, label_path in enumerate(glob.glob("simulated_data/Test/*_target.png")):
        filename = label_path.split("/")[-1]
        path = label_path.split(filename)[0]
        id = filename.split("_")[0]
        image_path=path+id+".png"

        image = Image.open(image_path).convert('L')
        X_test[i,:,:] = convert_tensor(image)
        
        label = Image.open(label_path).convert('L')
        Y_test[i,:,:] = convert_tensor(label)
    """    
    for i, label_path in enumerate(glob.glob("../WestCam/images_for_training/*_target.png")):
        filename = label_path.split("/")[-1]
        path = label_path.split(filename)[0]
        id = filename.split("_")[0]
        image_path=path+id+".png"

        image = Image.open(image_path).convert('L')
        X_train[i,:,:] = convert_tensor(image)

        label = Image.open(label_path).convert('L')
        Y_train[i,:,:] = convert_tensor(label)
    
    return X_train, Y_train#, X_test, Y_test

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=True, hflip_p=0.5, vflip_p=0.5):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.hflip_p=hflip_p
        self.vflip_p=vflip_p

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        rand_hflip = random.random()
        rand_vflip = random.random()
        vflip = transforms.RandomVerticalFlip(p=1)
        hflip = transforms.RandomHorizontalFlip(p=1)
        
        if rand_hflip<self.hflip_p:
            x = hflip(x)
            y = hflip(y)
        if rand_vflip<self.vflip_p:
            x = vflip(x)
            y = vflip(y)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

"""
tensor_x, tensor_y = load_images()
#transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5)])
train_dataset_vf = CustomTensorDataset(tensors=(tensor_x, tensor_y), transform=True,vflip_p=0.5,hflip_p=0.5)
print("aplsd")
my_dataloader = DataLoader(train_dataset_vf,shuffle=False,batch_size=2)
for i in range(4):
    for data in my_dataloader:
        print("inside")
        x,y=data
        print(x.shape)
        print(y.shape)
        plt.subplot(1,2,1)
        plt.imshow(y[0,:,:])
        plt.subplot(1,2,2)
        plt.imshow(x[0,:,:])
    plt.show()
"""