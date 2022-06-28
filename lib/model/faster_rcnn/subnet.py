
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch

class _VEAE(nn.Module): #view embd AutoEncoder 

    def __init__(self, in_dim = 49*32,encode_dim = 3,classify_dim = 32):
        super(_VEAE, self).__init__()
        self.encode = nn.Sequential(
                nn.Linear(in_dim, 256),nn.ELU(),nn.Dropout(0.4),
                nn.Linear(256, 32),nn.ELU(),nn.Dropout(0.4),
                nn.Linear(32, encode_dim), nn.ELU()
                        )
        self.decode = nn.Sequential(
                nn.Linear(encode_dim, 32),nn.ELU(),nn.Dropout(0.4),
                nn.Linear(32, 256), nn.ELU(),nn.Dropout(0.4),
                nn.Linear(256, in_dim), nn.Sigmoid()

                        )
        self.classify = nn.Sequential(
                nn.Linear(encode_dim, 32)
                        )

    def forward(self, x):
        x_encode = self.encode(x)
        x_rec = self.decode(x_encode)
        y = self.classify(x_encode)
        return x_encode, x_rec, y

class _VFAE(nn.Module): #view feature AutoEncoder 
    def __init__(self, in_dim = 49,encode_dim = 3,classify_dim = 32):
        super(_VFAE, self).__init__()
        self.encode = nn.Sequential(
                nn.Linear(in_dim, 32),nn.ELU(),nn.Dropout(0.4),
                nn.Linear(32, encode_dim), nn.ELU()
                        )
        self.decode = nn.Sequential(
                nn.Linear(encode_dim, 32),nn.ELU(),nn.Dropout(0.4),
                nn.Linear(32, in_dim), nn.Sigmoid()

                        )
        self.classify = nn.Sequential(
                nn.Linear(encode_dim, 32)
                        )

    def forward(self, x):
        x_encode = self.encode(x)
        x_rec = self.decode(x_encode)
        y = self.classify(x_encode)
        return x_encode, x_rec, y


class _SAConv2d(nn.Module): #switch attention conv

    def __init__(self, in_channels):
        super(_SAConv2d, self).__init__()
        self.switch = torch.nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            stride=1,
            bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(0)

    def forward(self, x):

        # switch
        #pdb.set_trace()
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)

        #out = switch * out_s + (1 - switch) * out_l
        return switch





class _SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(_SELayer, self).__init__()
        #import pdb;pdb.set_trace()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
        

    def forward(self, feature):
        #import pdb;pdb.set_trace()

        b, v, l, w, h  = feature.size()
        c = v*l
        feature = feature.view(b,c,w,h)
        response = self.avg_pool(feature).view(b,c)

        response_score = self.fc(response).view(b, c, 1, 1)
        feature = feature * response_score.expand_as(feature)
        return feature.view(b, v, l, w, h)

class _SE_Res_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(_SE_Res_Layer, self).__init__()
        #import pdb;pdb.set_trace()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
    def forward(self, feature,response):
        res = feature
        b, v, l, _, _  = feature.size()
        response_score = self.fc(response)
        response_pro = F.softmax(response_score,dim=1).view(b, v, 1, 1, 1)

        return feature * response_pro.expand_as(feature)+res ,response_score

class _RCNN(nn.Module):
    def __init__(self, input_dim,cls_output_dim, reg_output_dim):
        super(_RCNN, self).__init__()
        #import pdb;pdb.set_trace()
        self.fc_cls = nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cls_output_dim)
        )      
        self.fc_bbox = nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, reg_output_dim)
        )
        
        '''
        for p in self.parameters():
            p.requires_grad = False   
        '''
    def forward(self,inputs):
        cls_score = self.fc_cls(inputs)
        bbox_pred = self.fc_bbox(inputs)
        return cls_score,bbox_pred

class _PSCONV(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(_PSCONV, self).__init__()
        self.RCNN_cls_base = nn.Conv2d(in_channels=input_dim, out_channels=output_dim,
                                       kernel_size=1, stride=1, padding=0, bias=True)
        #self.RCNN_cls_base.weight.data.fill_(0)
        #self.RCNN_cls_base.bias.data.fill_(0)
        #self.rec = nn.ReLU(inplace=True)
        #import pdb;pdb.set_trace()
        '''
        for p in self.parameters():
            p.requires_grad = False
        '''
    def forward(self,inputs):
        vs_feat = self.RCNN_cls_base(inputs)

        return vs_feat

class _GECONV(nn.Module):
    def __init__(self, input_dim,output_dim):
        super(_GECONV, self).__init__()
        self.RCNN_general = nn.Conv2d(in_channels=input_dim, out_channels=output_dim,
                                       kernel_size=1, stride=1, padding=0, bias=False)
        #import pdb;pdb.set_trace()
        '''
        for p in self.parameters():
            p.requires_grad = False
        '''
    def forward(self,inputs):
        general_feat = self.RCNN_general(inputs)
        return general_feat


class _ViewSelect(nn.Module):
    def __init__(self, input_dim,view_output_dim):
        super(_ViewSelect, self).__init__()
        #import pdb;pdb.set_trace()
        self.fc_view = nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, view_output_dim)
        )      
        
        '''
        for p in self.parameters():
            p.requires_grad = False   
        '''
    def forward(self,inputs):
        view_score = self.fc_view(inputs)
        view_prob = F.softmax(view_score,dim = 1)
        return view_score,view_prob
