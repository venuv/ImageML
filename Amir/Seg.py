import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from pgpreprocess import pics

class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 100, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(2-1):
            self.conv2.append( nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(100) )
        self.conv3 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(2-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def Segment(model_address,im_address,pat_address):
    model = torch.load(model_address).eval()
    
    im = cv2.imread(im_address)
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')])/255.0 )
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, 100)
    _, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = im_target.reshape( [im.shape[0],im.shape[1]] ).astype( np.uint8 )
    return np.unique(im_target), im_target_rgb
    

