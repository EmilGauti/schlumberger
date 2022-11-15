import numpy as np
from PIL import Image
import glob
from torchvision import transforms     
import torch

def load_images(flip=False):
    test_images = []
    test_labels = []
    convert_tensor = transforms.ToTensor()
    for label_path in glob.glob("simulated_data/Test/*_target.png"):
        filename = label_path.split("/")[-1]
        path = label_path.split(filename)[0]
        id = filename.split("_")[0]
        image_path=path+id+".png"
        #image = imageio.imread(image_path)
        image = Image.open(image_path).convert('L')
        image = convert_tensor(image)
        #image = image[:,:,0]
        #image = np.moveaxis(image, -1,0)
        #label = imageio.imread(label_path)
        label = Image.open(label_path).convert('L')
        label = convert_tensor(label)
        test_images.append(image)
        test_labels.append(label)  

    X_test= torch.tensor(test_images)
    #Y_test= test_labels
    print(X_test.shape)
    #return X_train, Y_train, X_test, Y_test
load_images()