
# 
# ## within dataset class we would store the data as well as the label 
# ## function to initialize the dataset (include loading dataset) ## function for reading in images 
# ## function for getting labels ## function to get the size of dataset ## function to get an item within the dataset 
# # function to get train and test dataloader (an even split between each category) 
# # function and class to standarize dataset 
# # function to resize dataset #change file path to get image_0001.jpg path = os.path.realpath(__file__) dir = os.path.dirname(path) dir = dir.replace("CNN Fall 2023", "/CNN Fall 2023/jpg/") print(dir) os.chdir(dir) im = iio.imread('image_0001.jpg') #resize image (function) - resize all image to 1 resolution #im.resize(250, 250) 
# #data cleaning #have all the image into a dataset in Pytorch #all label goes into dataset 

import imageio as iio #imageio.v3 ??
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# from torch.utis.data import Dataset, Dataloader

class ImageSet():
    def __init__(self, current_foldername,dataset_foldername,transform=None):
        self.data = self.load_data(current_foldername,dataset_foldername)
        self.transform = transform
        mat_file = sio.loadmat('imagelabels.mat')
        self.sementic
        self.label = mat_file['labels']

    def __len__(self):
        return len(self.data)

    def __get_item__(self,idx):
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.label[idx])

    def load_data(current_foldername,dataset_foldername):
        # change current directory to dataset directory
        path = os.path.realpath(__file__)
        dir = os.path.dirname(path)
        dir = dir.replace(current_foldername, dataset_foldername)
        os.chdir(dir)

        X = [] #create an empty list
        for filename in os.listdir(dataset_foldername):
            X.append(iio.imread(dataset_foldername+'/'+filename))
        return X
    
    #crops all data
    #we want to crop to middle 
    def crop_images(self, res):
        for img in self.X:
        # img is a numpy array so we need to crop as a numpy array
            width, height = img.size #i think we need to use img.shape since img is np array

            if width > height:
                left = (img.width / 2) - (height / 2)
                right = (img.width / 2) + (height / 2)
                top = 0
                bottom = height
                img.crop((left, top, right, bottom))
                
            elif height > width:
                left = 0
                right = width
                top = (img.height / 2) - (width / 2)
                bottom = (img.height / 2) + (width / 2)
                img.crop((left, top, right, bottom))
        
    #resize all data
    def resize_image(self, res):
        for img in self.X:
            img.resize((res, res, 3)) # use pillow
    
    #get label of image given its index
    #need to check how data is ordered in dataset,
    #which labels=which flowers
    #def get_label(idx):


    #view a single image from 
    def view_data(idx):
        plt.figure(figsize=(8,8))
        plt.imshow(self.X(idx))
        plt.show()


# crop and resize function
# loading into dataloader
# data augmentation if needed


#to view image


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


# use imageio to convert png into numpy resolution of image is 250*250 np array will be 3*250*250 # dataset class 
# change file path to get image_0001.jpg
# path = os.path.realpath(__file__)
# dir = os.path.dirname(path)
# dir = dir.replace("CNN Fall 2023", "/CNN Fall 2023/jpg/")
# print(dir)
# os.chdir(dir)


#resize image (function) - resize all image to 1 resolution

#im.resize(250, 250)


#data cleaning




#have all the image into a dataset in Pytorch



#all label goes into dataset


