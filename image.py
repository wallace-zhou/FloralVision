
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
from torchvision import transforms
import math
# from torch.utis.data import Dataset, Dataloader


#for 10/31 i want you guys to work with only test_img folder
# we want to test extraction, crop and resize and have it in a dataset container
# we will then segement based on the setid
# then we will work begin on dataloader object

# segment
# x = image set
# y = flower classifications?



def get_loader(batch_size, task = "train"):
    #TODO: try different transformation
    preprocess = transforms.Compose([# Crop the center of the image
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),          # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],    # Mean values for the model
            std=[0.229, 0.224, 0.225]       # Standard deviation values
        )
    ])
    dataset = ImageSet(task, transform=preprocess)
    # dataset.X = dataset.X.transpose(0,3,1,2)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle = False)
    return loader




class ImageSet():
    def __init__(self, task = "train", transform=None):
        # self.data = self.load_data(current_foldername,dataset_foldername)
        mat_file = sio.loadmat('setid.mat')
        partition =[]
        if (task == "train"):
            partition = mat_file['trnid'][0]
        elif (task == "test"):
            partition = mat_file['tstid'][0]
        elif (task == "validation"):
            partition = mat_file['valid'][0]
        self.task = task
        self.transform = transform
        mat_file = sio.loadmat('imagelabels.mat')
        self.sementic = []
        self.label = mat_file['labels'][0]
        self.X, self.y = self.load_partition(partition)
        

    def __len__(self):
        return len(self.X)
        
    def __get_item__(self,idx):
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.label[idx])
    
    def load_all_data(self):
        x = np.array([])
        for filename in os.listdir('test_img'):
            x.append(iio.imread('test_img/'+filename))
        return x
    
    def load_partition(self, partition):
        X = []
        y = []
        missing = 0
        print("loading data")
        for id in partition:

            y.append(self.label[id]-1)
            id_string = str(id).zfill(5)
            try:
                img = iio.imread('jpg/image_'+id_string+'.jpg')
                img = self.crop_image(img)
                img = self.resize_image(img,224)
                X.append(img)
            except FileNotFoundError:
                missing+=1
        print("finish loading")
        return np.array(X), np.array(y)
            

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
    def crop_image(self,img):
        height = img.shape[0]
        width = img.shape[1] #i think we need to use img.shape since img is np array

        if (width > height):
            left = math.floor((width / 2) - (height / 2))
            right = math.floor((width / 2) + (height / 2))
            img = img[:,left:right,:]    
        elif (height > width):
            top = math.floor((height / 2) - (width / 2))
            bottom = math.floor((width / 2) + (height / 2))
            img = img[top:bottom,:,:]
        return img

    def resize_image(self, img, res):
        image_pil = Image.fromarray(img)

        # Resize the image to 224x224 using Pillow
        desired_shape = (res, res)
        resized_image_pil = image_pil.resize(desired_shape, Image.BILINEAR)  # You can choose the resampling method

        # Convert the Pillow image back to a NumPy array
        img = np.array(resized_image_pil) 
        return img
                
    def crop_images(self):
        # img is a numpy array so we need to crop as a numpy array
        for idx, img in enumerate(self.data):
            height = img.shape[0]
            width = img.shape[1]
            print(width, height) #i think we need to use img.shape since img is np array

            if (width > height):
                left = math.floor((width / 2) - (height / 2))
                right = math.floor((width / 2) + (height / 2))
                self.data[idx] = img[:,left:right,:]    
            elif (height > width):
                top = math.floor((height / 2) - (width / 2))
                bottom = math.floor((width / 2) + (height / 2))
                self.data[idx] = img[:,:,top:bottom]
            
        
    #resize all images 
    def resize_images(self, res):
        for idx, img in enumerate(self.data):
            image_pil = Image.fromarray(img)

            # Resize the image to 224x224 using Pillow
            desired_shape = (res, res)
            resized_image_pil = image_pil.resize(desired_shape, Image.BILINEAR)  # You can choose the resampling method

            # Convert the Pillow image back to a NumPy array
            self.data[idx] = np.array(resized_image_pil) 
            #check this out https://www.geeksforgeeks.org/python-pil-image-resize-method/#
    
    def __getitem__(self, idx):
        """Return (image, label) pair at index `idx` of dataset."""
        return self.transform(self.X[idx]), torch.tensor(self.y[idx]).long()
    
    #get label of image given its index
    def get_label(self,idx):
        return 0
        #TODO: given idx and class, what label is the idx
    #need to check how data is ordered in dataset,
    #which labels=which flowers
    #def get_label(idx):


    #view a single image from 
    def view_data(self,idx):
        plt.imshow(self.X[idx])
        plt.show()



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

if __name__ == "__main__":
    a = ImageSet(task = "train")
    a.view_data(idx = 0)
    loader = get_loader(32)
