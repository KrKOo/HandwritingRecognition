import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights, ViT_L_32_Weights, ViT_B_16_Weights
import dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(144256,256)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()
        self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(self.vgg.features.children()))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = nn.Flatten()

        embedding_size = 512
        # Additional metric learning layers
        self.fc1 = nn.Linear(512 * 7 * 7, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 256)
        self.norm = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Pass input through the pretrained VGG network
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        
        # Feature representation
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Additional metric learning layers
        x = self.fc2(x)
        
        return x


class MyTransformer(nn.Module):
    def __init__(self):
        super(MyTransformer, self).__init__()
        self.vit_model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        for param in self.vit_model.parameters():
            param.requires_grad = False
            
        # remove the clasification head of the vision transformer
        self.vit_model.heads = nn.Identity()
        
        self.additional_layers = nn.Sequential(
            nn.Linear(768, 256),  # Example additional layer
            nn.ReLU(),
            nn.Linear(256, 512)  # Example additional layer
        )
        
    def forward(self, x):
        x = self.vit_model(x)
        x = self.additional_layers(x)
        
        return x


