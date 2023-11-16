import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transform

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d()

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.flatten = nn.Flatten()

        self.fc1 = 
