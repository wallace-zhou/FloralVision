import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet50(weights='DEFAULT')
        for param in self.cnn.parameters():
            param.requires_grad = False
        input = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(input,102)

    def forward(self,x):
        return self.cnn(x)
    

