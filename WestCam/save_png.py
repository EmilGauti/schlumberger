import numpy as np
import matplotlib.pyplot as plt

dir="arrays/"
ind=350
arr = np.load(dir+"depth_slice_"+str(ind)+".npy")
arr_filt = np.load(dir+"depth_slice_filt_"+str(ind)+".npy")


fig=plt.figure()

ax = fig.add_subplot(2,2,1)
ax.imshow(arr,cmap="gray")
ax.set_title("original")
ax = fig.add_subplot(2,2,2)
ax.imshow(arr_filt,cmap="gray")
ax.set_title("filtered")
ax = fig.add_subplot(2,2,3)
ax.imshow(arr-arr_filt,cmap="gray")
ax.set_title("original - filtered")
ax = fig.add_subplot(2,2,4)
ax.imshow(arr_filt-arr,cmap="gray")
ax.set_title("filtered - original")


fig.suptitle("Image Nr. "+str(ind))
plt.savefig("tmp"+str(ind)+".png")
plt.show()

