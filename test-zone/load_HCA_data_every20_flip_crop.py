import numpy as np
from PIL import Image, ImageOps
import glob
from torchvision import transforms
import torchvision     
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import random
import cv2 as cv
from PIL import Image

def load_images(skip=20):
    test_images = []
    test_labels = []
    
    #X_test = torch.zeros([0,231,231], dtype=torch.float32)
    #Y_test = torch.zeros([0,231,231], dtype=torch.float32)
    X_train = torch.zeros([1,891,1055], dtype=torch.float32)
    Y_train = torch.zeros([1,891,1055], dtype=torch.float32)
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
    for i, label_path in enumerate(glob.glob("../Hca/Hca_masks/Hca_mask_*.png")):
        filename = label_path.split("/")[-1]
        #path = label_path.split(filename)[0]
        id = filename.split("_")[-1]
        id = id.split(".")[0]
        if int(id)%skip==0:
            image_path="../Hca/Hca_images/Hca_image_"+id+".png"

            image = Image.open(image_path).convert('L')
            image = np.array(image)
            clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
            image = clahe.apply(image)
            image_tensor = torch.tensor(image)
            image_tensor = torch.unsqueeze(image_tensor,0)
            X_train = torch.cat((X_train,image_tensor))
            #X_train[i,:,:] = torch.tensor(image)
            #X_train[i,:,:] = convert_tensor(image)

            label = Image.open(label_path).convert('L')
            label_tensor = convert_tensor(label)
            label_tensor = torch.unsqueeze(label_tensor,0)
            Y_train = torch.cat((Y_train, convert_tensor(label)))
    X_train_red = X_train[1:,:,:]
    Y_train_red = Y_train[1:,:,:]
    print(len(X_train_red),"images loaded from Hca")
    
    return X_train_red, Y_train_red#, X_test, Y_test

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=True, hflip_p=0, vflip_p=0):
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

        params = transforms.RandomCrop.get_params(x, output_size=(128, 128))
        x = transforms.functional.crop(x,*params)
        y = transforms.functional.crop(y,*params)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

"""
tensor_x, tensor_y = load_images()
tensor_x = tensor_x/255
tensor_y = tensor_y/255
plt.imshow(tensor_x[0,:,:],cmap="gray")
plt.show()


# Plot from val/test tensor

def generate_pic(tensor_x_val, tensor_y_val, im_nr):
    image = tensor_x_val[im_nr, :, :]
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    image_target = tensor_y_val[im_nr, :, :]
    image_target = image_target.detach().numpy()
    image = torch.squeeze(image)
    image = image.detach().numpy()
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(image_target, cmap="gray")
    plt.show()


# my_loaded_network = Net()
# my_loaded_network.load_state_dict(torch.load("models/model_12"))
#generate_pic(my_nn.to(device="cpu"), tensor_x_val, tensor_y_val, im_nr=5)

train_dataset_vf = CustomTensorDataset(tensors=(tensor_x, tensor_y), transform=True,vflip_p=0.5,hflip_p=0.5)
print("aplsd")
my_dataloader = DataLoader(train_dataset_vf,shuffle=False,batch_size=1)

for i, (x_test, y_test) in enumerate(my_dataloader):
    if i == 200:
    # x_test = x_test.to(device=device)
    # y_test = y_test.to(device=device)
        x_test = torch.unsqueeze(x_test, 1)
        target = y_test
        target = torch.squeeze(target)
        target = target.flatten()
        target = target.cpu().detach().numpy()
        generate_pic(x_test, y_test, im_nr=0)
  #  target = convToBinary(target)
"""