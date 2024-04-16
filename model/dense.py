import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torchvision.models as models

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.densenet161(weights='DEFAULT')
        for param in self.cnn.parameters():
            param.requires_grad = False
        input = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Linear(input,102)

    def forward(self,x):
        return self.cnn(x)
    

