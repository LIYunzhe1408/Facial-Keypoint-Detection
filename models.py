from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torchvision import models

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()

#         # input image : 1 x 224 x 224, grayscale squared images

#         self.conv1 = nn.Conv2d(1, 32, 4)  # 32*(4,4) filter ==> 221*221*32
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2, 2)  # pool (2,2) ==> 110*110*32
#         self.dropout1 = nn.Dropout(p=0.1)

#         # TODO: add more layers
#         self.conv2 = nn.Conv2d(32, 64, 3)       # 64 x 108 x 108
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(2, 2)         # 64 x 54 x 54
#         self.dropout2 = nn.Dropout(p=0.2)

#         self.conv3 = nn.Conv2d(64, 128, 3)      # 128 x 52 x 52
#         self.bn3 = nn.BatchNorm2d(128)
#         self.pool3 = nn.MaxPool2d(2, 2)         # 128 x 26 x 26
#         self.dropout3 = nn.Dropout(p=0.3)

#         self.conv4 = nn.Conv2d(128, 256, 3)     # 256 x 24 x 24
#         self.bn4 = nn.BatchNorm2d(256)
#         self.pool4 = nn.MaxPool2d(2, 2)         # 256 x 12 x 12
#         self.dropout4 = nn.Dropout(p=0.4)

#         self.fc1 = nn.Linear(256 * 12 * 12, 1000)
#         self.fc2 = nn.Linear(1000, 1000)
#         self.fc3 = nn.Linear(1000, 136)

#         I.xavier_uniform_(self.fc1.weight.data)
#         I.xavier_uniform_(self.fc2.weight.data)
#         I.xavier_uniform_(self.fc3.weight.data)

#     def forward(self, x):
        
#         # TODO: implement forward pass
#         x = self.pool1(nn.ReLU()(self.bn1(self.conv1(x))))
#         x = self.dropout1(x)

#         x = self.pool2(nn.ReLU()(self.bn2(self.conv2(x))))
#         x = self.dropout2(x)

#         x = self.pool3(nn.ReLU()(self.bn3(self.conv3(x))))
#         x = self.dropout3(x)

#         x = self.pool4(nn.ReLU()(self.bn4(self.conv4(x))))
#         x = self.dropout4(x)

#         x = x.view(x.size(0), -1)

#         x = nn.ReLU()(self.fc1(x))
#         x = nn.ReLU()(self.fc2(x))
#         x = self.fc3(x)

#         return x

class SimpleNet(nn.Module):
    """
    A deeper VGG-like face keypoint regressor.
    Input:  1 x 224 x 224  (grayscale)
    Output: N x 136        (e.g. 68 keypoints x 2 coords each)
    """
    def __init__(self, num_keypoints=136):
        super(SimpleNet, self).__init__()

        # Feature extractor (conv + batchnorm + relu + pool)
        # 1) 1 -> 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   # 224 -> 112
        )

        # 2) 32 -> 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   # 112 -> 56
        )

        # 3) 64 -> 128
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   # 56 -> 28
        )

        # 4) 128 -> 256
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   # 28 -> 14
        )

        # 5) 256 -> 512
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   # 14 -> 7
        )

        # 6) 512 -> 512
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   # 7 -> 3 (floor)
        )

        # After 6 pools: feature map is roughly 512 x 3 x 3 = 4608
        # Classifier / Regressor head:
        self.fc1 = nn.Linear(512*3*3, 2048)
        self.fc2 = nn.Linear(2048, num_keypoints)

        # Optional: Xavier init
        I.xavier_uniform_(self.fc1.weight)
        I.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Two FC layers
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)

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

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        n_inputs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(n_inputs, 136)

    def forward(self, x):
        x = self.resnet34(x)
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


class Resnet34Grayscale(nn.Module):
    def __init__(self):
        super(Resnet34Grayscale, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        # TODO: modify resnet18 to grayscale

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(resnet34.conv1.weight.sum(dim=1, keepdim=True))

        # Copy all layers except the first conv and fc
        self.backbone = nn.Sequential(*list(resnet34.children())[1:-1])

        # Fully connected layer for regression
        self.fc = nn.Linear(512, 136)


    def forward(self, x):
        x = self.conv1(x)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



from transformers import AutoImageProcessor, AutoModel
class DINOv2Keypoint(nn.Module):
    def __init__(self, model_name='facebook/dinov2-base', num_keypoints=68):
        super(DINOv2Keypoint, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size  # usually 768

        # Keypoint regression head
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_keypoints * 2)
        )

    def forward(self, x):
        # x: [B, 1, 224, 224] -> [B, 3, 224, 224]
        x = x.repeat(1, 3, 1, 1)
        outputs = self.backbone(pixel_values=x).last_hidden_state  # [B, num_patches, hidden]
        cls_token = outputs[:, 0]  # Use CLS token
        out = self.head(cls_token)
        return out

class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))

class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.upconv(x)))


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pool(x)).view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7, stride=7, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1, 1, 1)
        return self.activate(self.bn(self.convTrans(x)))


# Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downConv = DownConv(in_channels, out_channels)
        self.conv_block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downConv(x)
        return self.conv_block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = UpConv(in_channels, out_channels)
        self.conv_block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        return self.conv_block(x)

class UnconditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_hiddens: int,
    ):
        super().__init__()
        self.initial_conv = ConvBlock(in_channels, num_hiddens)

        self.down1 = DownBlock(num_hiddens, num_hiddens)  # 28x28 -> 14x14
        self.down2 = DownBlock(num_hiddens, num_hiddens * 2)  # 14x14 -> 7x7

        self.flatten = Flatten()
        self.unflatten = Unflatten(num_hiddens * 2, num_hiddens * 2)

        self.up2 = UpBlock(num_hiddens * 4, num_hiddens)
        self.up1 = UpBlock(num_hiddens * 2, num_hiddens)
        self.last_to_second = ConvBlock(num_hiddens * 2, num_hiddens)
        # self.final_conv = nn.Conv2d(num_hiddens, in_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(num_hiddens, 68, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2:] == (224, 224), "Expect input shape to be (224, 224)."

        x_init = self.initial_conv(x)
        x1 = self.down1(x_init)
        x2 = self.down2(x1)

        latent = self.flatten(x2)  # (B, num_hiddens*4, 1, 1)
        latent = self.unflatten(latent)  # (B, num_hiddens*4, 7, 7)
        # print("latent_unflattened: ", latent.shape)

        con = torch.cat([latent, x2], dim=1)  # Concatenation with downsampled feature map
        x = self.up2(con)  # (B, num_hiddens*2, 14, 14)


        con = torch.cat([x, x1], dim=1)  # Concatenation with initial feature map
        x = self.up1(con)  # (B, num_hiddens, 28, 28)


        con = torch.cat([x, x_init], dim=1)  # Concatenation with initial feature map
        x = self.last_to_second(con)  # (B, num_hiddens, 28, 28)
        x = self.final_conv(x)  # Final convolution

        return x
