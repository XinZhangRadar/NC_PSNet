# encoding: utf-8
from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# modified by Yicheng Liu
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval
import pdb;
# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

from geo_distance import *
# <<<< obsolete


class spacenet(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        #pdb.set_trace()
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        #self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        '''
        self._classes = ('__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor','plane')
         '''       
        #self._classes = ('__background__', 'plane')
        # self._classes = ('__background__', 'f-16','j-10','su-33ub','yf-22','e-2c','f-14','j-5','fa-18e')
        #self._classes = ('__background__','car','train','tvmonitor','diningtable','boat','aeroplane','bus','bicycle','chair','bottle','sofa','motorbike')
        #self._classes = ('__background__','aeroplane','ashtray','backpack','basket','bed','bench','bicycle','blackboard','boat','bookshelf','bottle','bucket','bus','cabinet','calculator','camera','can','cap','car','cellphone','chair','clock','coffee_maker','comb','computer','cup','desk_lamp','diningtable','dishwasher','door','eraser','eyeglasses','fan','faucet','filing_cabinet','fire_extinguisher','fish_tank','flashlight','fork','guitar','hair_dryer','hammer','headphone','helmet','iron','jar','kettle','key','keyboard','knife','laptop','lighter','mailbox','microphone','microwave','motorbike','mouse','paintbrush','pan','pen','pencil','piano','pillow','plate','pot','printer','racket','refrigerator','remote_control','rifle','road_pole','satellite_dish','scissors','screwdriver','shoe','shovel','sign','skate','skateboard','slipper','sofa','speaker','spoon','stapler','stove','suitcase','teapot','telephone','toaster','toilet','toothbrush','train','trash_bin','trophy','tub','tvmonitor','vending_machine','washing_machine','watch','wheelchair')
        self._classes = ('__background__','building')
        #self._classes =('__background__','desk_lamp','bicycle','helmet','aeroplane','car','faucet','boat','motorbike','bench','diningtable')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
        self.view_dict = {'1030010003D22F00':7,'1030010003993E00':6,'1030010002B7D800':5,'1030010002649200':4,'1030010003127500':3,'103001000307D800':2,'1030010003315300':1,
        '103001000392F600':0,'10300100023BC100':8,'1030010003CAF100':9,'10300100039AB000':10,'1030010003C92000':11,'103001000352C200':12,'1030010003472200':13,'10300100036D5200':14,
        '1030010003697400':15,'1030010003895500':16,'1030010003832800':17,'10300100035D1B00':18,'1030010003CCD700':19,'1030010003713C00':20,'10300100033C5200':21,'1030010003492700':22,
        '10300100039E6200':23,'1030010003BDDC00':24,'1030010003CD4300':25,'1030010003193D00':25}
        self.view_list = (-32,-29,-25,-21,-16,-13,-10,-7,8,10,14,19,23,27,30,34,36,39,42,44,46,47,49,50,52,53)
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        #return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit')
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb
        #pdb.set_trace()

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        try:
            tree = ET.parse(filename)
            objs = tree.findall('object')
        except:
            objs =  []

        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)
        
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        _view_id = index.split('_')[4]
        #poses = np.zeros((num_objs, 3), dtype=np.int32)

        

        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            '''
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            '''
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text) 
            if x1 < 0 :
                x1 = 0
            if y1 <0 :
                y1 = 0
            #pose      = obj.find('pose')
            
            
            #elevation = int(index.split('_')[2].split('r')[1])
            #azimuth = 0
            #distance = 1
            '''
            try:
                distance   = float(pose.find('distance').text)
                azimuth   = float(pose.find('azimuth').text)
                elevation = float(pose.find('elevation').text)
            except:
                import pdb; pdb.set_trace()
            '''
            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            #poses[ix, :] = [distance,azimuth,elevation]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
        #import pdb; pdb.set_trace()
        #import pdb;pdb.set_trace()
        overlaps = scipy.sparse.csr_matrix(overlaps)
        view, distance_gt = self.view_fliter(_view_id)
        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'affine':False,
                'seg_areas': seg_areas,
                'views':view,
                'distance':distance_gt
                }


    def view_fliter(self, _view_id):
        '''
            input: poses[ix,:]  = [distance,azimuth,elevation]
            return view[ix] = view ->[1,32]
        '''
        num_view = 26+1
        #num_view = 1+1
        #num_view = 1+1
        

        distance = np.zeros((num_view),dtype=np.float32)
 

        view, distance = self.view_label(_view_id,num_view)
        #pdb.set_trace()
        return view, distance
    def view_label(self, _view_id,num_view):

        view_label = self.view_dict[_view_id]
        azimuth = 0
        elevation = self.view_list[view_label]

        distance = np.zeros(num_view,dtype=np.float32)

        for view_int in range(len(self.view_list)) :
            
            mid_e = self.view_list[view_int]
            mid_a = 0
            distance[view_int] = getDistance(mid_e,mid_a,elevation,azimuth)
        
        #pdb.set_trace()
        distance[-1]=view_label
        return view_label, distance
    def view_label_18(self, _view_id,num_view):
        '''
        input: azimuth
               elevation
        out  : view label
        注意分类这里第一项要小于第二项而且第二项不能超过360 注意是个圆
        ''' 
        #a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal

        #a_intervals = list(([-180,-165],[-165,-105],[-105,-75],[-75,-15],[-15,15],[15,75],[75,105],[105,165],[165,180])) # object
        #pdb.set_trace()
        e_intervals = [[i,i+5] for i in range(-35,55,5)]

        #e_intervals = [[i,i+10] for i in range(-35,55,10)]


        azimuth = 0
        elevation = self.view_list[self.view_dict[_view_id]]

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

    def view_label_9(self, _view_id,num_view):
        '''
        input: azimuth
               elevation
        out  : view label
        注意分类这里第一项要小于第二项而且第二项不能超过360 注意是个圆
        ''' 
        #a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal

        #a_intervals = list(([-180,-165],[-165,-105],[-105,-75],[-75,-15],[-15,15],[15,75],[75,105],[105,165],[165,180])) # object
        #pdb.set_trace()
        e_intervals = [[i,i+10] for i in range(-35,55,10)]

        #e_intervals = [[i,i+10] for i in range(-35,55,10)]


        azimuth = 0
        elevation = self.view_list[self.view_dict[_view_id]]

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

    def view_label_5(self, _view_id,num_view):
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

        e_intervals = [[-35,-25],[-25,0],[0,25],[25,40],[40,55]]
        azimuth = 0
        elevation = self.view_list[self.view_dict[_view_id]]



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



    def view_label_1(self, _view_id,num_view):
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
        view_label = 0
        distance = np.zeros(num_view,dtype=np.float32)
        distance[-1] = view_label

        #e_intervals = [[i,i+10] for i in range(-35,55,10)]
  
        return view_label,distance

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        #pdb.set_trace();
        filedir = "/home/zhangxin/faster-rcnn.pytorch/VGG_evaluate/"
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir,thresh):
        ovthresh = thresh
        annopath = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC' + self._year,
            'ImageSets',
            'Main',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        cachedir = "/home/zhangxin/faster-rcnn.pytorch/VGG_evaluate/annotations_cache"
        aps = []
        # The PASCAL VOC metric changed in 2010

        use_07_metric = False

        #use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap, fp = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir,thresh,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.9f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap, 'fp':fp}, f)
        print('Mean AP = {:.9f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.9f}'.format(ap))
        print('{:.9f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir,thresh):

        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir,thresh)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = spacenet('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
