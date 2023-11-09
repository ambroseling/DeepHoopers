import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

class MLP(nn.Module):
  def __init__(self,num_features):
    super(MLP, self).__init__()
    self.layer1 = nn.Linear(num_features, 32,device="mps")
    self.layer2 = nn.Linear(32, 64,device="mps")
    self.layer3 = nn.Linear(64, 128,device="mps")
    self.layer4 = nn.Linear(128, num_features,device="mps")
    self.relu = nn.ReLU()

  def forward(self, x):
    activation1 = self.layer1(x)
    activation1 = self.relu(activation1)
    activation2 = self.layer2(activation1)
    activation2 = self.relu(activation2)
    activation3 = self.layer3(activation2)
    activation3 = self.relu(activation3)
    activation3 = self.layer4(activation3)
    return activation3