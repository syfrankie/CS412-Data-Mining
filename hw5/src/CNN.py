import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

class CNN(nn.Module):
    def __init__(self, in_channel, out_channel, num_classes, dropout=0.5):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1), # [in_channel, 30, 30]
        	nn.Conv2d(in_channel, out_channel, 4, stride=2, bias=False), # [out_channel, 13, 13]
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channel, out_channel*2, 4, stride=2, bias=False), # [out_channel*2, 5, 5]
            nn.BatchNorm2d(out_channel*2),
            nn.AdaptiveMaxPool2d((2, 2)) # [out_channel*2 , 2, 2]
        )
        self.fc = nn.Sequential(
            nn.Linear(out_channel*8, 64),
        	nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        	nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        out = self.fc(x)
        return out