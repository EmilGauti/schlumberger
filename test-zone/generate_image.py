import numpy as np
import matplotlib.pyplot as plt
import random

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def generate(x_dim, y_dim, period):
    my_dpi = 100
    arr = np.zeros((x_dim,y_dim))
    for i in range(20000):
        x=random.randint(0-10,x_dim+10)
        y=random.randint(0-10,x_dim+10)
        fwhm=10#random.random()*10+10
        arr=arr+makeGaussian(x_dim,fwhm=fwhm,center=[x,y])
    #arr=np.log(arr)
    arr = np.abs(arr - np.min(arr))/np.max(arr)
    noise = np.random.normal(0, .05, arr.shape)
    arr=arr+noise
    #arr = np.where(arr<0.1,arr+random.random()*1/arr,arr)
    print(arr)
    plt.imshow(arr, interpolation='none',cmap="gray",norm=None)
    plt.show()

generate(400,400,24*np.pi)

