import imageio as iio
import torch
import os
# from torch.utis.data import Dataset, Dataloader


# use imageio to convert png into numpy resolution of image is 250*250 np array will be 3*250*250
# dataset class
## within dataset class we would store the data as well as the label
## function to initialize the dataset (include loading dataset)
## function for reading in images
## function for getting labels
## function to get the size of dataset
## function to get an item within the dataset

# function to get train and test dataloader (an even split between each category)
# function and class to standarize dataset 
# function to resize dataset



#change file path to get image_0001.jpg
path = os.path.realpath(__file__)
dir = os.path.dirname(path)
dir = dir.replace("CNN Fall 2023", "/CNN Fall 2023/jpg/")
print(dir)
os.chdir(dir)

im = iio.imread('image_0001.jpg')

#resize image (function) - resize all image to 1 resolution

#im.resize(250, 250)


#data cleaning




#have all the image into a dataset in Pytorch



#all label goes into dataset


