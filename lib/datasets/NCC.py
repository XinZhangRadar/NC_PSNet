#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author : Liu Yicheng
# Date : 2019/8/15 
import numpy as np
import torch
import cv2
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import pdb
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import functools
from align_transform import Align
class surf_(object):
    def __init__(self,img1,img2):
        self.img1 = img2
        self.img2 = img1
    def surf_kp(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        suft = cv2.xfeatures2d_SURF.create()
        kp, des = suft.detectAndCompute(image, None)
        kp_image = cv2.drawKeypoints(gray_image, kp, None)
        return kp_image, kp, des

    def get_good_match(self,des1, des2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
        return good

    def suftImageAlignment(self):
        _, kp1, des1 = self.surf_kp(self.img1)
        _, kp2, des2 = self.surf_kp(self.img2)
        goodMatch = self.get_good_match(des1, des2)
        if len(goodMatch) > 4:
            ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ransacReprojThreshold = 4
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold);
            imgOut = cv2.warpPerspective(self.img2, H, (self.img1.shape[1],self.img1.shape[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return imgOut, H, status
class NCCloss():
    def __init__(self):
        self.image_size = 100
        self.image_f_size = 100

        self.num_classes =8
        self.DICT_PIXEL_MEANS = np.array([[[115,117.6, 115]]])
        self.dict_np = np.ones([24*(self.num_classes+1),self.image_f_size,self.image_f_size,3]).astype(np.uint8) #view,channel,w,h
        self.dict_com = np.ones([self.num_classes,self.image_f_size,self.image_f_size*24,3]).astype(np.uint8)
        
        #dict_data = torch.FloatTensor(1).cuda()
        #self.dict_data = Variable(dict_data)
        self.loss_1 = torch.zeros(24)
        
        #pdb.set_trace();
        dict_path = '/home/zhangxin/View-sensitive-conv/dictimage/'
        
        d = 0
        for c in range(self.num_classes+1):
            l = 0
            for wei in range(3):
                for jing in range(8):
                    if c==self.num_classes:

                        im =cv2.imread(dict_path+'background.jpg')
                        im = cv2.resize(im,(self.image_size,self.image_size),interpolation=cv2.INTER_CUBIC)
                        im = cv2.resize(im,(self.image_f_size,self.image_f_size),interpolation=cv2.INTER_CUBIC)
                        #im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                        self.dict_np[d,:,:,:] = im
                        d = d+1

                    else:
                        im =cv2.imread(dict_path+str(c+5)+'-'+str(wei+1)+'-'+str(jing+1)+'.jpg')
                        im = cv2.resize(im,(self.image_size,self.image_size),interpolation=cv2.INTER_CUBIC)
                        im = cv2.resize(im,(self.image_f_size,self.image_f_size),interpolation=cv2.INTER_CUBIC)
                        #im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
                        #dict_np[d,:,:,:] = np.transpose(im,(2,0,1))
                        self.dict_np[d,:,:,:] = im
                        #pdb.set_trace()
                        self.dict_com[c,:,l*self.image_size:(l+1)*self.image_size,:] = im
                        l = l+1
                        d = d+1
        #pdb.set_trace()
        #self.dict_np = dict_np - self.DICT_PIXEL_MEANS.reshape(3,1,1)
        #self.dict_tensor = torch.from_numpy(dict_np).cuda()
        #self.dict_data.data.resize_([24*(self.num_classes+1),64,64]).copy_(self.dict_tensor)
      
    def aHash(self,img):
        s=0
        hash_str=''
        #遍历累加求像素和
        for i in range(8):
            for j in range(8):
                s=s+img[i,j]
        #求平均灰度
        avg=s/64
        #灰度大于平均值为1相反为0生成图片的hash值
        for i in range(8):
            for j in range(8):
                if  img[i,j]>avg:
                    hash_str=hash_str+'1'
                else:
                    hash_str=hash_str+'0'            
        return hash_str

    
    def dHash(self,img):
        hash_str=''
        #每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(8):
            for j in range(8):
                if   img[i,j]>img[i,j+1]:
                    hash_str=hash_str+'1'
                else:
                    hash_str=hash_str+'0'
        return hash_str
    
    def cmpHash(self,hash1,hash2):
        n=0
        #hash长度不同则返回-1代表传参出错
        if len(hash1)!=len(hash2):
            return -1
        #遍历判断
        for i in range(len(hash1)):
            #不相等则n计数+1，n最终为相似度
            if hash1[i]!=hash2[i]:
                n=n+1
        return n




    def getMatchNum(self,matches,ratio):
        '''返回特征点匹配数量和匹配掩码'''
        matchesMask=[[0,0] for i in range(len(matches))]
        matchNum=0
        for i,(m,n) in enumerate(matches):
            if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
                matchesMask[i]=[1,0]
                matchNum+=1
        return (matchNum,matchesMask)
    def siftsim(self,img1,img2):


        comparisonImageList=[] #记录比较结果

        #创建SIFT特征提取器
        sift = cv2.xfeatures2d.SIFT_create() 
        #创建FLANN匹配对象
        FLANN_INDEX_KDTREE=0
        indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
        searchParams=dict(checks=50)
        flann=cv2.FlannBasedMatcher(indexParams,searchParams)

        kp1, des1 = sift.detectAndCompute(img1, None) #提取样本图片的特征
        kp2, des2 = sift.detectAndCompute(img2, None) #提取比对图片的特征
        matches=flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        (matchNum,matchesMask)=self.getMatchNum(matches,0.9) #通过比率条件，计算出匹配程度
        matchRatio=matchNum*100/len(matches)
        return matchRatio
    def template_matching(self,template,source):
        pdb.set_trace()
        img = source[:,:,0]
        img2 = img.copy()
        template = template[:,:,0]
        w, h = template.shape[::-1]

        # All the 6 methods for comparison in a list
        methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

        for meth in methods:
            img = img2.copy()
            method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv2.rectangle(img,top_left, bottom_right, 255, 2)

            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            pdb.set_trace()

            plt.show()

    def briefsim(self,img1,img2):
        
        al = Align(img1, img2, threshold=1)
        img1,img2,M = al.align_image()
        cv2.imwrite('m.jpg',img2)
        pdb.set_trace()

        #surf = surf_(img1,img2)
        #img,H,status = surf.suftImageAlignment()
        '''


        comparisonImageList=[] #记录比较结果

        surf = cv2.xfeatures2d.SURF_create()
        kp1 = surf.detect(img1)
        kp2 = surf.detect(img2)

        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp1, des1 = brief.compute(img1, kp1)
        kp2, des2 = brief.compute(img2, kp2)


        #创建FLANN匹配对象
        bf=cv2.BFMatcher()
        matches=bf.match(des1.astype(np.uint8),des2.astype(np.uint8),k=2)
        #matches=sorted(matches,key=lambda x:x.distance)
        #img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:5],None,flags=2)
        #plt.imshow(img3),plt.show()
        
        (matchNum,matchesMask)=self.getMatchNum(matches,0.75) #通过比率条件，计算出匹配程度
        matchRatio=matchNum*100/len(matches)
        '''
        
        return 0#matchRatio


    def loss(self, image, cls):
        """
            inputs:
                x : original picture [Batch,3,W,H]
                position_x : position of the roi [Batch,5]
                y:dictionary picture [24*(C+1),3,W,H]
                position_y : position of the dict need to be pooled [Batch,5]
            return:
                NCCloss 
        """
        eps = 0.000001

        dic = self.dict_np[cls*24:(cls+1)*24,:,:]

        #dic = self.dict_data.data[cls*24:(cls+1)*24,:,:]#torch.Size([24, 3, 256, 256]) 
        # 将图片大小变成一样的
        # cv2.imwrite("/home/zhangxin/View-sensitive-conv/lib/datasets/test_dic{}.png".format(cls+1),np.rollaxis( np.array(self.dict_data.data[(cls+4)*24+1,:,:,:]),0,3)  )
        w,h = image.shape[0],image.shape[1]
        if w >= h :
            scale = dic.shape[1]/w
            image = cv2.resize(image, (dic.shape[1],int(scale*h)), interpolation=cv2.INTER_CUBIC)
            #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            crop_left = int((dic.shape[2]-int(scale*h))/2)
            crop_right = crop_left + int(scale*h)
            dic = dic[:,:,crop_left:crop_right,:]
        else:
            scale = dic.shape[2]/h
            image = cv2.resize(image, (dic.shape[2],int(scale*w)), interpolation=cv2.INTER_CUBIC)
            #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            crop_left = int((dic.shape[1]-int(scale*w))/2)
            crop_right = crop_left + int(scale*w)
            dic = dic[:,crop_left:crop_right,:,:]
        
        #roi = torch.from_numpy(image).cuda().float()
        #pdb.set_trace()
        #roi_hash = self.aHash(roi)
        

        

        for i in range(dic.shape[0]):
            #pdb.set_trace()

            #self.loss_1[i] = self.siftsim(image,dic[i])
            self.loss_1[i] = self.briefsim(image,dic[9])
            #dic_i_hash = self.aHash(dic[i])
            #self.loss_1[i] = self.cmpHash(roi_hash,dic_i_hash)
        angle = torch.argmax(self.loss_1)
        '''
        source = self.dict_com[cls]
        self.template_matching(image,source)
        '''




        return angle