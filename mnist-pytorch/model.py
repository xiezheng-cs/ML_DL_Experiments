import torch
import torch.nn.modules as nn
import torch.nn.functional as F
# define LeNet model

class LeNet(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LeNet, self).__init__()
        self.conv = nn.container.Sequential(
            nn.conv.Conv2d(in_dim, 6, 5, stride=1, padding=0),  #1*28*18-->6*24*24
            nn.activation.ReLU(True),
            nn.pooling.MaxPool2d(2, 2),                   #6*24*24-->6*12*12
            
            nn.conv.Conv2d(6, 16, 5, stride=1, padding=0),    #6*12*12-->16*8*8
            nn.activation.ReLU(True), 
            nn.pooling.MaxPool2d(2, 2))                  #16*8*8-->16*4*4

        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120), 
            nn.Linear(120, 84), 
            nn.Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.softmax(out)
        return out
