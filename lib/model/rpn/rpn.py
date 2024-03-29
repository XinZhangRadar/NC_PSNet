#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author : Liu Yicheng( Modified )
# Date : 2019/9/14 
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss
from model.rpn.bbox_transform import bbox_overlaps_batch

import numpy as np
import math
import pdb
import time
import matplotlib.pyplot as plt

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        #pdb.set_trace()
        '''
        for p in self.parameters():
            p.requires_grad = False
        '''
    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        #pdb.set_trace();

        batch_size = base_feat.size(0)
        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=False)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal view layer
        # rpn_view_pred = self.RPN_view_pred(rpn_conv1)
        # rpn_view_pred_reshape = self.reshape(rpn_view_pred, self.view_size)
        # rpn_view_prob_reshape = F.softmax(rpn_view_pred_reshape, 1)
        # rpn_view_prob = self.reshape(rpn_view_prob_reshape, self.nc_view_out)
        
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # rois,score, view= self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, 
        #                         rpn_view_prob.data,
        #                         im_info, cfg_key))
        #rois,score= self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, None,
        rois= self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, None, 
                                im_info, cfg_key))
        #TODO:
        '''
        1.FC计算roi视角分类 √
        1.计算图中目标的视角的GT（GMM）
        2.匹配roi的视角GT
        3.计算roi视角loss
        '''

        '''
        overlaps = bbox_overlaps_batch(rois, gt_boxes)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        X = max_overlaps.cpu().numpy();
        Y = score[0].cpu().detach().numpy();
        plt.scatter(X, Y)
        plt.show()
        '''
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        # self.rpn_loss_view = 0
        v_gt_label = 0


        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            

            # compute classification loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            # rpn_view_pred = rpn_view_pred_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.view_size)
            rpn_label = rpn_data[0].view(batch_size, -1)
            
            ###Get 256d (positive sample + negtive sampe)
           

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1)) #positive
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            # rpn_view_pred = torch.index_select(rpn_view_pred.view(-1,self.view_size), 0, rpn_keep) 
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            
            

            # view_gt = view_label1
            # offset = torch.arange(0,batch_size)*view_gt.size(1)
            # overlaps_indeces = rpn_data[-1].view(batch_size, -1)
            # overlaps_indeces = overlaps_indeces + offset.view(batch_size,1).type_as(overlaps_indeces)

            # overlaps_indeces = torch.index_select(overlaps_indeces.view(-1), 0, rpn_keep.data)
            # overlaps_indeces = Variable(overlaps_indeces.long())

            # view_label = overlaps_indeces.new(overlaps_indeces.size()).fill_(0)
            # #pdb.set_trace()
            # view_label[rpn_label.nonzero().squeeze()] = view_gt.view(-1)[overlaps_indeces[rpn_label.nonzero().squeeze()]].long()

            # v_gt_label = view_label
            
            
            


            # self.rpn_loss_view = F.cross_entropy(rpn_view_pred, view_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            

            
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
            
            

        # return rois,self.rpn_loss_cls, self.rpn_loss_box,self.rpn_loss_view,score,view, v_gt_label
        return rois, self.rpn_loss_cls, self.rpn_loss_box#, score
