#!/usr/bin/python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
#from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
sys.path.append('/home/zhangxin/Matlab/R2016b/extern/engines/python/build/lib/')
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.io
from PIL import Image, ImageDraw
from model.utils.hungarian import Hungarian
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random
from model.faster_rcnn.geo_distance import *

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  #daraset
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='objectnet3d', type=str)
  #config
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  #net
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  #set
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  #load dir
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',                
                      #default="/srv/share/jyang375/models")
                      default="./data/pretrained_model")
  #image dir 
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  #cuda
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  #multiple gpus
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  #class_agnostic bbox regression
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  #parallel
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  #check session
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  #check epoch
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  #check point
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  #batch size                    
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  #visualization                    
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  #webcam ID
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)

  args = parser.parse_args()
  return args
#learning rate
lr = cfg.TRAIN.LEARNING_RATE
#momentum
momentum = cfg.TRAIN.MOMENTUM
#weight_decay
weight_decay = cfg.TRAIN.WEIGHT_DECAY
def view_label_5( _view_id,num_view):
    '''
    input: azimuth
           elevation
    out  : view label
    注意分类这里第一项要小于第二项而且第二项不能超过360 注意是个圆
    ''' 
    #a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal

    #a_intervals = list(([-180,-165],[-165,-105],[-105,-75],[-75,-15],[-15,15],[15,75],[75,105],[105,165],[165,180])) # object
    #pdb.set_trace()
    #e_intervals = [[i,i+10] for i in range(-35,55,10)]

    #e_intervals = [[i,i+10] for i in range(-35,55,10)]
    view_dict = {'1030010003D22F00':7,'1030010003993E00':6,'1030010002B7D800':5,'1030010002649200':4,'1030010003127500':3,'103001000307D800':2,'1030010003315300':1,
    '103001000392F600':0,'10300100023BC100':8,'1030010003CAF100':9,'10300100039AB000':10,'1030010003C92000':11,'103001000352C200':12,'1030010003472200':13,'10300100036D5200':14,
    '1030010003697400':15,'1030010003895500':16,'1030010003832800':17,'10300100035D1B00':18,'1030010003CCD700':19,'1030010003713C00':20,'10300100033C5200':21,'1030010003492700':22,
    '10300100039E6200':23,'1030010003BDDC00':24,'1030010003CD4300':25,'1030010003193D00':25}
    view_list = (-32,-29,-25,-21,-16,-13,-10,-7,8,10,14,19,23,27,30,34,36,39,42,44,46,47,49,50,52,53) 
    e_intervals = [[-35,-25],[-25,0],[0,25],[25,40],[40,55]]
    azimuth = 0
    elevation = view_list[view_dict[_view_id]]



    distance = np.zeros(num_view,dtype=np.float32)

    for view_int in range(len(e_intervals)) :
        interval_l = e_intervals[view_int][0] 
        interval_r = e_intervals[view_int][1]
        if (elevation >= interval_l) and (elevation < interval_r) :
            view_label = view_int
        
        mid_e = (interval_l+interval_r )/2
        mid_a = 0
        distance[view_int] = getDistance(mid_e,mid_a,elevation,azimuth)
    
    #pdb.set_trace()
    distance[-1]=view_label
    return view_label, distance
view_size = 26 + 1
def view_label_9( _view_id,num_view):
    '''
    input: azimuth
           elevation
    out  : view label
    注意分类这里第一项要小于第二项而且第二项不能超过360 注意是个圆
    ''' 
    #a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal
    #
    #a_intervals = list(([-180,-165],[-165,-105],[-105,-75],[-75,-15],[-15,15],[15,75],[75,105],[105,165],[165,180])) # object
    #pdb.set_trace()
    view_dict = {'1030010003D22F00':7,'1030010003993E00':6,'1030010002B7D800':5,'1030010002649200':4,'1030010003127500':3,'103001000307D800':2,'1030010003315300':1,
    '103001000392F600':0,'10300100023BC100':8,'1030010003CAF100':9,'10300100039AB000':10,'1030010003C92000':11,'103001000352C200':12,'1030010003472200':13,'10300100036D5200':14,
    '1030010003697400':15,'1030010003895500':16,'1030010003832800':17,'10300100035D1B00':18,'1030010003CCD700':19,'1030010003713C00':20,'10300100033C5200':21,'1030010003492700':22,
    '10300100039E6200':23,'1030010003BDDC00':24,'1030010003CD4300':25,'1030010003193D00':25}
    view_list = (-32,-29,-25,-21,-16,-13,-10,-7,8,10,14,19,23,27,30,34,36,39,42,44,46,47,49,50,52,53) 
    e_intervals = [[i,i+10] for i in range(-35,55,10)]

    #e_intervals = [[i,i+10] for i in range(-35,55,10)]


    azimuth = 0
    elevation = view_list[view_dict[_view_id]]

    distance = np.zeros(num_view,dtype=np.float32)

    for view_int in range(len(e_intervals)) :
        interval_l = e_intervals[view_int][0] 
        interval_r = e_intervals[view_int][1]
        if (elevation > interval_l) and (elevation <= interval_r) :
            view_label = view_int
        
        mid_e = (interval_l+interval_r )/2
        mid_a = 0
        distance[view_int] = getDistance(mid_e,mid_a,elevation,azimuth)
    
    #pdb.set_trace()
    distance[-1]=view_label
    return view_label, distance

def view_label( _view_id,num_view):
    '''
    input: azimuth
           elevation
    out  : view label
    注意分类这里第一项要小于第二项而且第二项不能超过360 注意是个圆
    ''' 
    #a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal

    #a_intervals = list(([-180,-165],[-165,-105],[-105,-75],[-75,-15],[-15,15],[15,75],[75,105],[105,165],[165,180])) # object
    
    #e_intervals = list(([0,11],[11,25],[25,35],[35,40],[40,45],[45,48],[48,52],[52,55]))

    #pdb.set_trace()
    view_dict = {'1030010003D22F00':7,'1030010003993E00':6,'1030010002B7D800':5,'1030010002649200':4,'1030010003127500':3,'103001000307D800':2,'1030010003315300':1,
    '103001000392F600':0,'10300100023BC100':8,'1030010003CAF100':9,'10300100039AB000':10,'1030010003C92000':11,'103001000352C200':12,'1030010003472200':13,'10300100036D5200':14,
    '1030010003697400':15,'1030010003895500':16,'1030010003832800':17,'10300100035D1B00':18,'1030010003CCD700':19,'1030010003713C00':20,'10300100033C5200':21,'1030010003492700':22,
    '10300100039E6200':23,'1030010003BDDC00':24,'1030010003CD4300':25,'1030010003193D00':25}
    view_list = (-32,-29,-25,-21,-16,-13,-10,-7,8,10,14,19,23,27,30,34,36,39,42,44,46,47,49,50,52,53)    
    view_label = view_dict[_view_id]
    azimuth = 0
    elevation = view_list[view_label]

    distance = np.zeros(num_view,dtype=np.float32)

    for view_int in range(len(view_list)) :
        
        mid_e = view_list[view_int]
        mid_a = 0
        distance[view_int] = getDistance(mid_e,mid_a,elevation,azimuth)
    
    #pdb.set_trace()
    distance[-1]=view_label
    return view_label, distance

def view_fliter(_view_id):
    '''
        input: poses[ix,:]  = [distance,azimuth,elevation]
        return view[ix] = view ->[1,32]
    '''
    num_view = view_size
    #pdb.set_trace()

    distance = np.zeros((num_view),dtype=np.float32)


    view, distance = view_label(_view_id,num_view)
    return view, distance
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  #
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)
def visualize(iteration, error, X, Y, ax):
    '''
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='red', label='Target')
    ax.scatter(Y[:,0] ,  Y[:,1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)
    '''
    return X,Y

if __name__ == '__main__':
  #pdb.set_trace()
  
  args = parse_args()
  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)
  #os.environ["CUDA_VISIBLE_DEVICES"] = "7"

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  #pdb.set_trace()
  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  #input_dir = args.load_dir
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    # 'psnet_eccv_256_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    'psnet_L30_V26_1_10_10880.pth')
    # 'faster_rcnn_fc_layer_50_view_12{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
  # pascal_classes = np.asarray(['__background__',
                      #  'aeroplane', 'bicycle', 'bird', 'boat',
                      #  'bottle', 'bus', 'car', 'cat', 'chair',
                      #  'cow', 'diningtable', 'dog', 'horse',
                      #  'motorbike', 'person', 'pottedplant',
                      #  'sheep', 'sofa', 'train', 'tvmonitor'])

  # pascal_classes = np.asarray(['__background__','desk_lamp','bicycle','helmet','aeroplane','car','faucet','boat','motorbike','bench','diningtable'])
  #pascal_classes = np.asarray(['__background__','car','train','tvmonitor','diningtable','boat','aeroplane','bus','bicycle','chair','bottle','sofa','motorbike'])
  #pascal_classes = np.asarray(['__background__', 'f-16','j-10','su-33ub','yf-22','e-2c','f-14','j-5','fa-18e'])
  # pascal_classes = np.asarray(['__background__','aeroplane','ashtray','backpack','basket','bed','bench','bicycle',\
  #                              'blackboard','boat','bookshelf','bottle','bucket','bus','cabinet','calculator','camera',\
  #                              'can','cap','car','cellphone','chair','clock','coffee_maker','comb','computer','cup','desk_lamp',\
  #                              'diningtable','dishwasher','door','eraser','eyeglasses','fan','faucet','filing_cabinet','fire_extinguisher',\
  #                              'fish_tank','flashlight','fork','guitar','hair_dryer','hammer','headphone','helmet','iron','jar','kettle','key','keyboard','knife','laptop','lighter','mailbox','microphone','microwave','motorbike','mouse','paintbrush','pan','pen','pencil','piano','pillow','plate','pot','printer','racket','refrigerator','remote_control','rifle','road_pole','satellite_dish','scissors','screwdriver','shoe','shovel','sign','skate','skateboard','slipper','sofa','speaker','spoon','stapler','stove','suitcase','teapot','telephone','toaster','toilet','toothbrush','train','trash_bin','trophy','tub','tvmonitor','vending_machine','washing_machine','watch','wheelchair'])

  pascal_classes = np.asarray(['__background__','building'])
  # initilize the network here.
  #args.class_agnostic = False  
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic,k=30)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  #pdb.set_trace()
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
  cfg.POOLING_MODE = 'align'

  print('load model successfully!')
  #pdb.set_trace()

  # pdb.set_trace()

  #print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  distance = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    distance = distance.cuda()
  #pdb.set_trace()
  # make variable
  with torch.no_grad():
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    distance = Variable(distance)
  
    if args.cuda > 0:
      cfg.CUDA = True
  
    if args.cuda > 0:
      fasterRCNN.cuda()
  
    fasterRCNN.eval()
  
    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True
  
    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0 :
      cap = cv2.VideoCapture(webcam_num)
      num_images = 0
    else:
      imglist = os.listdir(args.image_dir)
      num_images = len(imglist)
  
    print('Loaded Photo: {} images.'.format(num_images))
    image_size = np.zeros(2);
    im_save = [];
    pos_allimg = [];
    area_allimg =[];
    #pdb.set_trace();
    while (num_images > 0):
        total_tic = time.time()
        if webcam_num == -1:
          num_images -= 1
  
        # Get image from the webcam
        if webcam_num >= 0:
          if not cap.isOpened():
            raise RuntimeError("Webcam could not open. Please check connection.")
          ret, frame = cap.read()
          im_in = np.array(frame)
        # Load the demo image
        else:

          _view_id = imglist[num_images].split('_')[4]

          view, distance_pt = view_fliter(_view_id)
          print(view)
          im_file = os.path.join(args.image_dir, imglist[num_images])
          #im = cv2.imread(im_file)
          im_in = np.array(imread(im_file))
          #pdb.set_trace
          # im_in = np.array(Image.open(im_file))
        #pdb.set_trace()
          if len(im_in.shape) == 2:
            im_in = im_in[:,:,np.newaxis]
            im_in = np.concatenate((im_in,im_in,im_in), axis=2)
        # rgb -> bgr
        #pdb.set_trace()
          im_in = im_in[:,:,::-1]
        im = im_in
        # im = im_in # for plt

        #image list,image pyrimd scale
        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        #image list
        im_blob = blobs
        #image shape
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        #chang image type:numpy->tensor
        im_data_pt = torch.from_numpy(im_blob)
        distance_pt = torch.from_numpy(distance_pt)
        #chang image dim order
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        #chang image information type:numpy->tensor
        im_info_pt = torch.from_numpy(im_info_np)
        #assign data to placehoder
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 6).zero_()
        num_boxes.data.resize_(1).zero_()
        distance.data.resize_(1, view_size).copy_(distance_pt)
  
        # pdb.set_trace()
        det_tic = time.time()
        # faster-RCnn, produce bbox,bbox class score,bbox regression result,
        #torch.with_no_grad():
        with torch.no_grad():
          rois,cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box,\
          RCNN_loss_cls, RCNN_loss_bbox,\
          rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes,distance)
        #class scores and boxes
        #pdb.set_trace()
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]
        args.class_agnostic = True
  
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data.cuda()
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if args.class_agnostic:
                  if args.cuda > 0:

                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
  
                  box_deltas = box_deltas.view(1, -1, 4)
              else:
                  if args.cuda > 0:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  else:
                      box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                  box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))
  
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
  
        pred_boxes /= im_scales[0]
  
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        #print(pred_boxes) 
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        #pdb.set_trace()
        #imshow result
        if vis:
            im2show = np.copy(im)

          # Bounding-box colors
      
      
        # cmap = plt.get_cmap("tab20b")
        # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        # bbox_colors = random.sample(colors, len(pascal_classes))
                  
 
        # plt.figure()
        # fig, ax = plt.subplots(1)
        # ax.imshow(im2show)
        
        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:

              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              #combine class score and boxes
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
              #order the boxes by class scores
              cls_dets = cls_dets[order]
              #pdb.set_trace();
              keep = nms(cls_dets, cfg.TEST.NMS, force_cpu= not cfg.USE_GPU_NMS)
              #print(len(keep))
              cls_dets = cls_dets[keep.view(-1).long()]

              if vis:
                im2show,pos,area = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
                ##########################


                dets = cls_dets.cpu().numpy()

                # for i in range(np.minimum(1000, dets.shape[0])):
                #     im_p  = np.copy(im2show);
                #     bbox  = tuple(int(np.round(x)) for x in dets[i, :4])
                #     score = dets[i, -1]
                #     # thresh = 0.6
                #     if score > thresh:
                #       box_w = bbox[2] - bbox[0]
                #       box_h = bbox[3] - bbox[1]

                #       color = colors[1]#bbox_colors[j]
 
                #       bbox1 = patches.Rectangle((bbox[0], bbox[1]), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")

                #       ax.add_patch(bbox1)
                      
                #       plt.text(
                #           bbox[0], 
                #           bbox[1],
                #           s = '',#s =  str(score),#pascal_classes[j],#str(score), #pascal_classes[j] + ' ' + str(score) ,
                #           color="white",
                #           verticalalignment="top",
                #           bbox={"color": color, "pad": 0},
                #       )
                      



        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
 
        # result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
        # plt.savefig(result_path, bbox_inches="tight", pad_inches=0.0)
        # plt.close()
        #print(pos_allimg)
        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        #position = cls_dets
        #pdb.set_trace()
        if webcam_num == -1:
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(num_images + 1, len(imglist), detect_time, nms_time))
            sys.stdout.flush()
  
        if vis and webcam_num == -1:
            # cv2.imshow('test', im2show)
            # cv2.waitKey(0)
            result_path = os.path.join(args.image_dir, imglist[num_images][:-4] + "_det.jpg")
            cv2.imwrite(result_path, im2show)

        
        
   