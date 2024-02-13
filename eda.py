import imageio.v3 as iio #imageio.v3 ??
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import math
import pdb

pdb.set_trace()
def crop_images(img):
    # img is a numpy array so we need to crop as a numpy array
    height = img.shape[0]
    width = img.shape[1]
    print(width, height) #i think we need to use img.shape since img is np array

    if (width > height):
        left = math.floor((width / 2) - (height / 2))
        right = math.floor((width / 2) + (height / 2))
        img = img[:,left:right,:]
        
    elif height > width:
        top = math.floor((height / 2) - (width / 2))
        bottom = math.floor((width / 2) + (height / 2))
        img = img[:,:,top:bottom]

    plt.imshow(img)
    plt.show()

img = np.array(iio.imread('jpg/image_00002.jpg'))
# img.dtype
print(img.shape)
crop_images(img)


            