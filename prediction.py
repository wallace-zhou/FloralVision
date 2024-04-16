import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import pytorch_lightning as pl
from torch.nn.functional import softmax
from train import FlowerModule
import numpy as np
import sys
import os
import imageio as iio
from torchvision import transforms
from model import dense
import pdb
from PIL import Image
import matplotlib.pyplot as plt

NAMES = np.load('label.npy')
cnn = dense.Dense()
additional_args = {
    "model": cnn
    # Add any other arguments here
}
# Load the Lightning module from the checkpoint
model = FlowerModule.load_from_checkpoint('prediction.ckpt', **additional_args)
model.eval()
def list_files(directory):
    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files

directory_path = "prediction_images"
files_in_directory = list_files(directory_path)
print("Files in directory:")
for file in files_in_directory:
    print(file)

imageFile = input("What is the image file name: ")
while(imageFile != 'exit'):
    path = 'prediction_images/'+imageFile
    img = iio.v2.imread(path)
    if(img.shape[2] == 4):
        img = img[:,:,:3]
    transformed_img = Image.fromarray(img)
    size = min(img.shape[0],img.shape[1])
    preprocess = transforms.Compose([
        
        transforms.ToTensor(),# Crop the center of the image
        transforms.CenterCrop(size),
        transforms.Resize((224,224)),
            # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],    # Mean values for the model
            std=[0.229, 0.224, 0.225]       # Standard deviation values
        )
    ])
    transformed_img = preprocess(transformed_img)
    transformed_img = transformed_img.to(torch.device('cuda'))
    transformed_img = transformed_img.unsqueeze(0)
    with torch.no_grad():
        y_hat = model(transformed_img)
    prob = torch.nn.functional.softmax(y_hat[0],dim = 0)
    category = torch.argmax(prob).item()
    category = NAMES[category]
    print("The predicted flower category is:", category)
    plt.figure(figsize=(6,8))
    plt.imshow(img)
    plt.title('Predicted Class: %s' % category)
    plt.axis('off')
    plt.show()
    imageFile = input("What is the image file name: ")
### TODO: load the labels
#         load the model
#         load image transform the image
#         inference with the image
#         convert label to semantic