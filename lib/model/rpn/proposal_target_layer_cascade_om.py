from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _ProposalTargetLayer_OM(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer_OM, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes,cls_scores):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        #pdb.set_trace()

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)
        cls_scores = torch.cat([cls_scores, gt_boxes[:,:,0]], 1)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)

        pe_rois_per_image = int(np.round(cfg.TRAIN.PE_FRACTION * rois_per_image))
        pe_rois_per_image = 1 if pe_rois_per_image == 0 else pe_rois_per_image

        ph_rois_per_image = int(np.round(cfg.TRAIN.PH_FRACTION * rois_per_image))
        ph_rois_per_image = 1 if ph_rois_per_image == 0 else ph_rois_per_image








        labels, rois, bbox_targets, bbox_inside_weights,overlaps_indece_batch,pos_easy_num,pos_hard_num,neg_hard_num = self._sample_rois_pytorch(
            all_rois, gt_boxes, pe_rois_per_image,ph_rois_per_image,
            rois_per_image, self._num_classes,cls_scores)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights,overlaps_indece_batch,pos_easy_num,pos_hard_num,neg_hard_num

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes,  pe_rois_per_image,ph_rois_per_image, rois_per_image, num_classes,cls_scores):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        #pdb.set_trace()

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        overlaps_indece = gt_assignment;

        #max_cls_scores = torch.max(cls_scores, 2)
        #pdb.set_trace()




        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,4].contiguous().view(-1).index((offset.view(-1),)).view(batch_size, -1)

        overlaps_indece_batch = overlaps_indece.new(batch_size, rois_per_image).fill_(100);
        
        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            pos_easy = torch.nonzero((max_overlaps[i] >= cfg.TRAIN.FG_THRESH) & (cls_scores[i] >= cfg.TRAIN.FG_THRESH)).view(-1)
            pos_hard = torch.nonzero((max_overlaps[i] >= cfg.TRAIN.FG_THRESH) &(cls_scores[i] <= cfg.TRAIN.FG_THRESH) ).view(-1)
            neg_hard =  torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO) ).view(-1)
            
            pos_easy_num = pos_easy.numel()
            pos_hard_num = pos_hard.numel()
            neg_hard_num = neg_hard.numel()

            if pos_easy_num > 0 and pos_hard_num > 0 and neg_hard_num > 0:
                pos_easy_rois_per_img = min(pe_rois_per_image, pos_easy_num)
                rand_num = torch.from_numpy(np.random.permutation(pos_easy_rois_per_img)).type_as(gt_boxes).long()
                pos_easy = pos_easy[rand_num[:pos_easy_rois_per_img]]

                pos_hard_rois_per_img = min(ph_rois_per_image, pos_hard_num)
                rand_num = torch.from_numpy(np.random.permutation(pos_hard_rois_per_img)).type_as(gt_boxes).long()
                pos_hard = pos_hard[rand_num[:pos_hard_rois_per_img]]   

                neg_hard_rois_per_img = rois_per_image - pos_easy_rois_per_img - pos_hard_rois_per_img
                rand_num = np.floor(np.random.rand(neg_hard_rois_per_img) * neg_hard_num)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                neg_hard = neg_hard[rand_num]


            elif pos_easy_num > 0 and pos_hard_num == 0 and neg_hard_num > 0:

                pe_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
                pos_easy_rois_per_img = min(pe_rois_per_image, pos_easy_num)
                rand_num = torch.from_numpy(np.random.permutation(pos_easy_rois_per_img)).type_as(gt_boxes).long()
                pos_easy = pos_easy[rand_num[:pos_easy_rois_per_img]]

                pos_hard_rois_per_img = 0


                neg_hard_rois_per_img = rois_per_image - pos_easy_rois_per_img - pos_hard_rois_per_img
                rand_num = np.floor(np.random.rand(neg_hard_rois_per_img) * neg_hard_num)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                neg_hard = neg_hard[rand_num]

            elif pos_easy_num > 0 and pos_hard_num == 0 and neg_hard_num == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) * pos_easy_num)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                pos_easy = pos_easy[rand_num]

                pos_easy_rois_per_img = rois_per_image
                pos_hard_rois_per_img = 0
                neg_hard_rois_per_img = 0

            elif pos_easy_num > 0 and pos_hard_num > 0 and neg_hard_num == 0:
                pe_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
                pos_easy_rois_per_img = min(pe_rois_per_image, pos_easy_num)
                rand_num = torch.from_numpy(np.random.permutation(pos_easy_rois_per_img)).type_as(gt_boxes).long()
                pos_easy = pos_easy[rand_num[:pos_easy_rois_per_img]]

                neg_hard_rois_per_img = 0
                
                pos_hard_rois_per_img = rois_per_image - pos_easy_rois_per_img - neg_hard_rois_per_img
                rand_num = np.floor(np.random.rand(pos_hard_rois_per_img) * pos_hard_num)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                pos_hard = pos_hard[rand_num]


            elif pos_easy_num == 0 and pos_hard_num == 0 and neg_hard_num > 0:
                rand_num = np.floor(np.random.rand(rois_per_image) * neg_hard_num)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                neg_hard = neg_hard[rand_num]
                neg_hard_rois_per_img = rois_per_image

                pos_easy_rois_per_img = 0
                pos_hard_rois_per_img = 0





            elif pos_easy_num == 0 and pos_hard_num > 0 and neg_hard_num == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) * pos_hard_num)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                pos_hard = pos_hard[rand_num]
                pos_hard_rois_per_img = rois_per_image

                pos_easy_rois_per_img = 0
                neg_hard_rois_per_img = 0





            elif pos_easy_num == 0 and pos_hard_num > 0 and neg_hard_num > 0:
                ph_rois_per_image = int(np.round(0.5 * rois_per_image))
                pos_hard_rois_per_img = min(ph_rois_per_image, pos_hard_num)
                rand_num = torch.from_numpy(np.random.permutation(pos_hard_rois_per_img)).type_as(gt_boxes).long()
                pos_hard = pos_hard[rand_num[:pos_hard_rois_per_img]]


               
                pos_easy_rois_per_img = 0

                try:
                    neg_hard_rois_per_img = rois_per_image - pos_easy_rois_per_img - pos_hard_rois_per_img
                    rand_num = np.floor(np.random.rand(neg_hard_rois_per_img) * neg_hard_num)
                    rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                    neg_hard = neg_hard[rand_num]
                except:
                    pdb.set_trace();

            else :
                raise ValueError("pos_easy_num = 0, pos_hard_num = 0 and neg_hard_num = 0, this should not happen!")


   

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([pos_easy, pos_hard,neg_hard], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])
            overlaps_indece_batch[i].copy_(overlaps_indece[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            fg_rois_per_this_image = pos_easy_rois_per_img+pos_hard_rois_per_img


            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
                #overlaps_indece_batch[i][fg_rois_per_this_image:] = 0.5


            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])

        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        #pdb.set_trace()


        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights,overlaps_indece_batch,pos_easy_num,pos_hard_num,neg_hard_num
