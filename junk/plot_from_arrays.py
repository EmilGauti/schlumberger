import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
def plot_array(im_nr=569):
    path="/Users/emilgautifridriksson/Documents/GitHub/schlumberger/WestCam/arrays/"
    arr = np.load(path+"depth_slice_"+str(im_nr)+".npy")
    arr_filt = np.load(path+"depth_slice_filt_"+str(im_nr)+".npy")
    

    plt.imshow(arr,cmap="gray")
    plt.axis('off')
    plt.savefig("image.png")
    plt.show()
    plt.imshow(arr_filt,cmap="gray")
    plt.axis('off')
    plt.savefig("image_filt.png")
    plt.show()
    plt.imshow(arr-arr_filt,cmap="gray")
    plt.axis('off')
    plt.savefig("image_diff.png")
    plt.show()

    image = Image.open("image_diff.png").convert('L')
    image = np.array(image)
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
    arr_hist = clahe.apply(image)
    plt.imshow(arr_hist,cmap="gray")
    plt.axis('off')
    plt.savefig("image_diff_hist.png")
    plt.show()
plot_array()