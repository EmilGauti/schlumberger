import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask_donut(mask,min,max):
    rows,cols = mask.shape
    crow,ccol = int(rows/2) , int(cols/2)
    for i in range(rows):
        for j in range(cols):
            dist=np.sqrt((crow-i)**2+(ccol-j)**2)
            if dist>min and dist<max:
                mask[i,j]=0
    return mask

img = cv2.imread('sample.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)
bs=1
mask=np.ones((rows,cols))
mask = mask_donut(mask,0,15)
mask = mask_donut(mask,30,500)

fshift = np.where(mask==0,0,fshift)
#fshift[crow-bs:crow+bs, ccol-bs:ccol+bs] = 0

magnitude_spectrum=np.where(fshift==0,0,magnitude_spectrum)
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back, cmap = 'gray')
plt.title('After Filtering'), plt.xticks([]), plt.yticks([])
plt.show()