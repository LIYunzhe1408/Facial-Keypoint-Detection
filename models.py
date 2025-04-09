from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torchvision import models

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        # input image : 1 x 224 x 224, grayscale squared images

        self.conv1 = nn.Conv2d(1, 32, 4)  # 32*(4,4) filter ==> 221*221*32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # pool (2,2) ==> 110*110*32
        self.dropout1 = nn.Dropout(p=0.1)

        # TODO: add more layers

        self.fc3 = nn.Linear(1000, 136)

        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        
        # TODO: implement forward pass

        return x




class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class Resnet18Grayscale(nn.Module):
    def __init__(self):
        super(Resnet18Grayscale, self).__init__()
        
        # TODO: modify resnet18 to grayscale

    def forward(self, x):
        x = self.resnet18(x)
        return x
