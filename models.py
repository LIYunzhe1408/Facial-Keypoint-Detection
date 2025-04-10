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
        self.conv2 = nn.Conv2d(32, 64, 3)       # 64 x 108 x 108
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)         # 64 x 54 x 54
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 3)      # 128 x 52 x 52
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)         # 128 x 26 x 26
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 3)     # 256 x 24 x 24
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)         # 256 x 12 x 12
        self.dropout4 = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(256 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)

        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        
        # TODO: implement forward pass
        x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = self.pool4(nn.ReLU()(self.bn4(self.conv4(x))))
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)

        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)

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
        resnet18 = models.resnet18(pretrained=True)
        # TODO: modify resnet18 to grayscale

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(resnet18.conv1.weight.sum(dim=1, keepdim=True))

        # Copy all layers except the first conv and fc
        self.backbone = nn.Sequential(*list(resnet18.children())[1:-1])

        # Fully connected layer for regression
        self.fc = nn.Linear(512, 136)


    def forward(self, x):
        x = self.conv1(x)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
