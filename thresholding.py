import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('seismic.png',0)
# global thresholding
ret1,th1 = cv.threshold(img,160,255,cv.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=155)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding"]
for i in range(2):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.savefig("otsuThresholding.png")
