#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib
 
matplotlib.use('AGG')
 
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import cv2
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.layer_util import *
from model.utils.net_utils import _smooth_l1_loss,_smooth_l1_view_loss,_crop_pool_layer, _affine_grid_gen, _affine_theta
from model.faster_rcnn.subnet import _RCNN,_PSCONV,_GECONV,_SAConv2d
import sys
import time
import pdb,pickle
sys.path.append('/home/zhangxin/faster-rcnn.pytorch/PreciseRoIPooling/pytorch/prroi_pool')

#from prroi_pool import PrRoIPool2D
from model.rpn.bbox_transform import bbox_overlaps_batch
import matplotlib.pyplot as plt
#from model.stn.stn import STN
import os
from model.faster_rcnn.RebalanceLoss import PerspectiveRebalance



class _fasterRCNN(nn.Module):
    '''faster RCNN '''
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
  

        self.view_size = 5+1
        self.view_size_fg = 5
    

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes, self.view_size)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.RCNN_Net = _RCNN(self.k*cfg.POOLING_SIZE**2,self.n_classes,4)
        self.PS_Net = _PSCONV(1024,self.view_size_fg * self.k)

        self.KLloss = nn.KLDivLoss(reduce=True)

        self.pTildeResoures = '/home/zhangxin/View-sensitive-conv/lib/datasets/pTilde_32.npy'
        self.viewRebalanceLoss = PerspectiveRebalance(classNum=32, gamma=0.5, pTildeResoures= self.pTildeResoures, temperature = 20, lambda_ = 0.6, loss_type = 3)
        self.if_show = False


    def forward(self, im_data, im_info, gt_boxes1, num_boxes, distance=0,epoch=0):

        import pdb;pdb.set_trace()
        self.batch_size = im_data.size(0)
        #avg_pool = PrRoIPool2D(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        im_info = im_info.data
        gt_boxes1 = gt_boxes1.data
        num_boxes = num_boxes.data
        # Backbone: feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        # feed base feature map tp RPN to obtain rois
        if self.training:
            view_label = gt_boxes1[:,:,5].contiguous()
            gt_boxes = gt_boxes1[:,:,0:5].contiguous()
            distance = distance.data
        else :
            view_label = gt_boxes1[:,5].contiguous()
            gt_boxes = gt_boxes1[:,0:5].contiguous()

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, view_label)
   
        
        base_feat = self.RCNN_conv_new(base_feat)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, view_label, distance)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, overlaps_indece_batch, roi_v_label, distance_gt = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            roi_v_label = Variable(roi_v_label.view(-1).long())#view_add
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            distance_gt  = Variable(distance_gt.view(-1, distance_gt.size(2)),requires_grad=False)


        else:
            rois_label = None
            roi_v_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

            distance_gt = None
        rpn_loss_view = 0

        rois = Variable(rois)

        name = 'align'


        
        vs_feat = self.PS_Net(base_feat)


        if self.if_show:
            vs = vs_feat.view(vs_feat.shape[0], self.view_size_fg, 50, base_feat.shape[2], base_feat.shape[3])
            bb = vs.sum(2).detach().cpu().numpy()
            self.draw_features(4,8,bb,'view1.jpg')
        
        

 
        import pdb;pdb.set_trace()


        pooled_feat = self.RCNN_roi_align(vs_feat, rois.view(-1, 5))
        pooled_feat = pooled_feat.view(pooled_feat.shape[0], self.view_size_fg, self.k, cfg.POOLING_SIZE, cfg.POOLING_SIZE)
        pooled_light = pooled_feat.sum(1)


        if self.training:

            index_fg = self.select_layer(roi_v_label)
            pooled_feat_fg = torch.index_select(pooled_feat, dim=0, index = index_fg)
            #pooled_feat_bg = torch.index_select(pooled_feat, dim=0, index = index_bg)
            
            
            distance_fg = torch.index_select(distance_gt[:,:-1], dim=0, index = index_fg)
            dis_score_fg = self.Gauss(distance_fg)

            dis_score_fg = torch.div(dis_score_fg,dis_score_fg.sum(1).view(index_fg.shape[0],1)).cuda()


        pooled_light = pooled_light.view(pooled_light.size(0),-1)

        cls_score,bbox_pred = self.RCNN_Net(pooled_light_sum)
    

        cls_prob = F.softmax(cls_score, dim = 1)


        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_select = 0
        RCNN_loss_fg_bg = 0



        # create loss of view, cls and bbox
        if self.training:
            # classification loss
            #import pdb;pdb.set_trace()
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label) # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            # select loss input:non-zero view roi feature map [B,3],one hot [B,3]
            #_,index_fg = self.select_layer(pooled_feat, roi_v_label)
            #import pdb;pdb.set_trace()
            #pooled_cls_fg = torch.index_select( pooled_feat, dim=0, index = index_fg)

            roi_response = self.view_response(pooled_feat_fg)
            
            #RCNN_loss_select = self.viewRebalanceLoss(roi_response, dis_score_fg, roi_v_label[index_fg]-1)
            
            #roi_response = F.softmax(roi_response,dim=1)

            #RCNN_loss_select = _smooth_l1_view_loss(roi_response,dis_score_fg)
 
            
            #RCNN_loss_select = _smooth_l1_view_loss(roi_response_cls[0].view(-1,33),dis_score[0].view(-1,33))
            #rois_view_label = roi_v_label[index_fg]-1
            
            roi_response_log = F.log_softmax(roi_response,dim=1)
            RCNN_loss_select = self.KLloss(roi_response_log,dis_score_fg)
            
            


            #print('RCNN_loss_select')
            #print(RCNN_loss_select)
            #print('Compute')
            #print(roi_response[0])
            #print('gt')
            #print(dis_score_fg[0])            
            '''
            pooled_bbox_fg = torch.index_select( pooled_feat_loc, dim=0, index = index_fg)
            roi_response_bbox = self.view_response(pooled_bbox_fg)
            #rois_view_label = roi_v_label[index_fg]-1
            roi_response_bbox = F.log_softmax(roi_response_bbox,dim=1)            
            '''






            # + self.KLloss(roi_response_bbox,dis_score_fg.cuda())#F.cross_entropy(roi_response, rois_view_label)
            #self.KLloss(roi_response_cls[0],dis_score[0])
    


        cls_prob = cls_prob.view(self.batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(self.batch_size, rois.size(1), -1)
        rcnn_rpn_score = cls_prob.clone()
        #import pdb;pdb.set_trace()


        '''
        if not self.training:
            rcnn_rpn_score[0] = (score[0].view(-1,1)*cls_prob[0])
            return rois, rcnn_rpn_score, bbox_pred, rpn_loss_cls, rpn_loss_bbox, rpn_loss_view, RCNN_loss_cls, RCNN_loss_bbox,RCNN_loss_select, rois_label,name
        '''
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox,RCNN_loss_select, rois_label


    def CrossEntropy(self, scores, target):
        '''
            INPUTS
                scores   NxQxXxY     feature map
                target   NxQxXxY     ground truth with Gaussian smoothed 
                weight   Nx1xXxY     rebalance coefficients
            OUTPUTS
                Loss     N     weighted loss 

        '''
        #pdb.set_trace()
        logProbs = scores.log()
        lossReducedClass = torch.gather(logProbs,1,target.view(target.size(0),-1))
        loss = -lossReducedClass.mean()
        return loss


    def _init_weights(self):

        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        #normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def draw_features(self,width,height,x,savename):
        #pdb.set_trace()
        #tic=time.time()
        fig = plt.figure(figsize=(16,16))
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
        for i in range(width*height):
            plt.subplot(width,height, i + 1)
            plt.axis('off')
            # plt.tight_layout()
            img = x[0, i, :, :]
            pmin = np.min(x)
            pmax = np.max(x)
            img = (img - pmin) / (pmax - pmin + 0.000001)*255
            img=img.astype(np.uint8)
            #img=cv2.applyColorMap(img, cv2.COLORMAP_COOL)
            img=cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
            #cv2.imwrite("{}_view_ROI.jpg".format(i),img)
            img = img[:, :, ::-1]
            plt.imshow(img)
            #pdb.set_trace()
            #cv2.imwrite("{}_view.jpg".format(i),img)

            #print("{}/{}".format(i,width*height))
            #pdb.set_trace()
        fig.savefig(savename, dpi=100)
        fig.clf()
        plt.close()

    def to_onehot(self,input, dim ):
        '''
        Generate input's onehot code  
        '''
        max_idx = torch.argmax(input, dim, keepdim=True)
        one_hot = torch.FloatTensor(input.shape).cuda()
        one_hot.zero_()
        one_hot.scatter_(dim, max_idx, 1)
        return one_hot, max_idx

    def select_layer(self,index_all):
        u'''
            Select angles channel and background of feature map.
            就是我们的函数就是先分出背景前景
            背景部分直接全部3个view的channel保存
            前景部分通过label选择channel 
            通过batch 来选择这样就可以与label相同
            两种输出方式：
                1· 输出和原输入一致的output 只是V：3->1
                2· 输出分开一部分是fg 另一部分是 bg
            inputs :
                x (result after roi_pooling)    : input feature maps [B, V, C, W, H]
                index : input index of which viewpoint should be taken in every roi [B,1]
                Note: the parameters with same name are equal 
            returns :
                mask : Selected feature maps 
        '''
        #import pdb;pdb.set_trace()
        idx_fg = torch.nonzero(index_all.ge(1))[:,0]
        #idx_bg = torch.nonzero(index_all.lt(1))[:,0]    
        return idx_fg


    def getDistance_all(self,featuremap,index):
        azimuth_gt   = torhc.fmod((index - 1),8)
        elevation_gt = torch.floor(torch.div(index-1, 8, out=None))


    def view_response(self,x):
        r'''
            Compute the view response on roi feature map. 
            inputs  :
                x : input feature maps [B, V, C, W, H]
            returns :
                response : view_response [B,V]
        '''
        #pdb.set_trace()
        B, V, C, W, H = x.size()
        x = x.view(B,V,C,-1)
        x_mean0=torch.mean(x,dim=3,keepdim=False) #[B,V,C]
        x_max0 = torch.mean(x_mean0,dim = 2, keepdim=False)
        #response = F.softmax(x_max0, dim = 1)
        # import pdb; pdb.set_trace()
        return x_max0

    def Gauss(self,x):
        #import pdb;pdb.set_trace()
        b = np.pi/(4*3)
        eu = (x/b) * (x/b)
        k = 1/(np.sqrt(2*np.pi)*b)
        f = np.exp(-eu/2)

        return f
    def select_loss(self,pooled_light,index_fg,roi_v_label,sigma=1.0, dim=[1]):
        r'''
            input: pooled_light: feature map after roi pooling [B,C*V,W,H]
                   index_fg: index of front ground in feature map 
                   roi_v_label: ground truth labels of view for every position object
                   sigma: ?
                   dim: ?
            return:
                   selected loss using L1 smooth loss to evaluate 

        '''

        pooled_fg = torch.index_select( pooled_light, dim=0, index = index_fg)

        roi_response = self.view_response(pooled_fg)
        #import pdb; pdb.set_trace()
        roi_view_oh = torch.zeros(roi_response.size(0), self.view_size_fg).scatter_(1, roi_v_label[index_fg].unsqueeze(1).cpu()-1, 1)

        sigma_2 = sigma ** 2
        select_diff = roi_response - roi_view_oh.cuda()
        abs_select_diff = torch.abs(select_diff)
        smoothL1_sign = (abs_select_diff < 1. / sigma_2).detach().float()
        in_loss = torch.pow(select_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_select_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        RCNN_loss_select = in_loss

        for i in sorted(dim, reverse=True):
          RCNN_loss_select = RCNN_loss_select.sum(i)
        RCNN_loss_select = RCNN_loss_select.mean()

        return RCNN_loss_select