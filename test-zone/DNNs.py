import torch
import torch.nn as nn
class BasicNet_old(nn.Module):
    def __init__(self):
      super(BasicNet_old, self).__init__()
      self.conv_stack =nn.Sequential(
        nn.Conv2d(1,32,3,stride=1,padding=1),
        nn.MaxPool2d((2,2),stride=2),
        nn.ReLU(),
        nn.Conv2d(32,128,3,stride=1,padding=1),
        nn.MaxPool2d((2,2),stride=2),
        nn.ReLU()
      )
      self.deconv_stack = nn.Sequential(
        nn.ConvTranspose2d(128,128,2,stride=2,padding=0),
        nn.Conv2d(128,32,3,stride=1,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32,32,2,stride=2,padding=0),
        nn.Conv2d(32,1,1,stride=1,padding=0),
        nn.Sigmoid()
      )
    def forward(self, x):
      # Use the rectified-linear activation function over x
      #x_clone = x.clone()
      output=self.conv_stack(x)
      output=self.deconv_stack(output)
      #output=output+x_clone
      # Apply softmax to x
      #output = F.log_softmax(x, dim=1)
      return output
class BasicNet(nn.Module):
    def __init__(self):
      super(BasicNet, self).__init__()
      self.conv_stack =nn.Sequential(
        nn.Conv2d(1,32,3,stride=1,padding=1),
        nn.MaxPool2d((2,2),stride=2),
        nn.ReLU(),
        nn.Conv2d(32,64,3,stride=1,padding=1),
        nn.MaxPool2d((2,2),stride=2),
        nn.ReLU()
      )
      self.deconv_stack = nn.Sequential(
        nn.ConvTranspose2d(64,64,2,stride=2,padding=0),
        nn.Conv2d(64,64,3,stride=1,padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64,64,2,stride=2,padding=0),
        nn.Conv2d(64,1,1,stride=1,padding=0),
        nn.Sigmoid()
      )
    def forward(self, x):
      # Use the rectified-linear activation function over x
      #x_clone = x.clone()
      output=self.conv_stack(x)
      output=self.deconv_stack(output)
      #output=output+x_clone
      # Apply softmax to x
      #output = F.log_softmax(x, dim=1)
      return output