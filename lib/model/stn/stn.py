import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import model.stn.stn_config as cfg
#from model.spp.SPP_layer import SPPLayer
import pdb

class STN(nn.Module):
    def __init__(self, mode='stn',in_channel = 512 ,out_channel = 512,pool_level=7):
        assert mode in ['stn', 'cnn']


        super(STN, self).__init__()
        #pdb.set_trace()
        self.mode = mode
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.pool_level=pool_level
        self.local_net = LocalNetwork(in_channel = self.in_channel ,pool_level = self.pool_level)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        '''
        self.fc = nn.Sequential(
            nn.Linear(in_features=height // 4 * width // 4 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(in_features=1024, out_features=10)
        )
        '''

    def forward(self, img):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w), (b,)
        '''
        batch_size = img.size(0)
        #pdb.set_trace()
        if self.mode == 'stn':
            transform_img = self.local_net(img)
            #img = transform_img
        else:
            transform_img = None

        #conv_output = self.conv(img).view(batch_size, -1)
        #predict = self.fc(conv_output)
        return transform_img#, predict


class LocalNetwork(nn.Module):
    def __init__(self,in_channel = 512 ,pool_level = 4):
        super(LocalNetwork, self).__init__()
        self.in_channel = in_channel

        self.pool_level = pool_level

        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channel * self.pool_level*self.pool_level,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(0.7),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))
        #self.spp = SPPLayer(4)


        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)
        #pdb.set_trace()
        height = img.size(2)
        width = img.size(3)



        #vec = self.spp(img);

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, self.in_channel, height, width)))
        img_transform = F.grid_sample(img, grid)

        return img_transform


if __name__ == '__main__':
    net = LocalNetwork()

    x = torch.randn(1, 1, 40, 40) + 1
    net(x)

