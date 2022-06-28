#pTilde
import numpy as np
import sklearn.neighbors as neighbors
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import os,shutil,sys,pdb
import xml.etree.ElementTree as ET
from collections import Counter
import math
from functools import reduce
from geo_distance import getDistance

def progress(count, total, status=''):
    # status_string="Processing | ( "+str(i+1)+"/"+str(len(name))+")"
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    if count == total-1:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, 100, '%', status))
        sys.stdout.write("\n")
    else:
        sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', status))
        
    sys.stdout.flush()

class GaussianSmoothedProbGenerator(object):

    def __init__(self, sigma=5, numNearestPoint=10,quantizedViewPointPath: str='', viewMap=None, noSmoothing:bool=False):
        if viewMap is None :
            viewMap = np.load(quantizedViewPointPath)
        self.sigma = sigma    
        metric = self.distanceMetric
        self.nearestNbr4Smooth = neighbors.NearestNeighbors(\
                                    n_neighbors=numNearestPoint,\
                                    metric=metric,\
                                    algorithm='ball_tree').fit(viewMap)
        self.nearestNbr = neighbors.NearestNeighbors(n_neighbors=1,metric=metric,\
                                   algorithm='ball_tree').fit(viewMap)
        self.viewMap = viewMap
        self.Q = viewMap.shape[0] #num of view/perspective
        self.noSmoothing = noSmoothing

    def gaussianSmooth(self, probs):
        
        (dists,inds) = self.nearestNbr4Smooth.kneighbors(self.viewMap)
        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts, axis=1, keepdims=True)
        pTilde = wts*probs[inds] # Weighting
        pTilde = pTilde.sum( axis=1)
        pTilde = pTilde/np.sum(pTilde) # re-normalized
        return pTilde

    def weightedAllDataPoint(self, dataPath:str=None, outputPath:str=None):

        pts = self.readData(dataPath)
        # pts: array [[azimuth, elevation],...]
        probs = self.originProbs(pts)
        # probs: array [n_view, probs of each view]
        if self.noSmoothing:
            pTilde = probs
        else:
            pTilde = self.gaussianSmooth(probs)
        # probs: array [n_view, smoothed probs of each view]
        np.save(outputPath,pTilde)

    def originProbs(self, pts: np.array):

        (dists,inds) = self.nearestNbr.kneighbors(pts)
        # inds array, shape (n_queries, n_neighbors)
        inds = list(inds.squeeze(axis=1)) # n_neighbors= 1
        self.counter=Counter(inds)
        probs = np.zeros(self.Q)
        indxs ,values = list(self.counter.keys()), list(self.counter.values())
        import pdb;pdb.set_trace()
        probs[indxs] = values
        probs = probs/np.sum(probs) # normalization
        return probs
    
    def readData(self, XMLPath:str=None):
        
        if XMLPath is None:
            # XMLPath='/data/zhangxin/ObjectNet3D/ObjectNet3D/Annotations/'
            XMLPath='/data/zhangxin/Pascal3D+/VOCdevkit/VOC2012/Annotations/'
            #im_path='/data3/gujie/data_/RSMatch/NewFinetune/JPEGImage/'
        filenamess=os.listdir(XMLPath)
        filenames=[]
        for name in filenamess:
            name=name.replace('.xml','')
            filenames.append(name)
        recs = {}
        pts = []

        unViewLabel = 0
        for i, name in enumerate(filenames):
            try:
                recs[name]=self.parse_obj(XMLPath, name+ '.xml' )
            except:
                unViewLabel+=1
                continue
            for object in recs[name]:
                pts.append(np.array(object['view']))
            status_string="Processing | ( "+str(i+1)+"/"+str(len(filenames))+")"
            progress(i,len(filenames),status_string)
        print("unViewLable: ",unViewLabel)
        pts = np.stack(pts)
        return pts

    def parse_obj(self, xml_path, filename):
        tree=ET.parse(xml_path+filename)
        objects=[]
        for obj in tree.findall('object'):
            obj_struct={}
            obj_struct['name']=obj.find('name').text
            pose      = obj.find('pose')
            # distance   = float(pose.find('distance').text)
            azimuth   = float(pose.find('azimuth').text)
            elevation = float(pose.find('elevation').text)
            label  = self.view_label(azimuth,elevation)
            obj_struct['view']=label
            objects.append(obj_struct)
        return objects
    
    def view_label(self, azimuth, elevation):
        return [azimuth, elevation]

    def distanceMetric(self, x, y):
        return getDistanceNp(x,y)
        # return np.sqrt(sum((x-y)**2))


def rad(d):
    return d * np.pi / 180.0
 
def getDistanceNp(x,y):
    '''
        inputs 
            x: array [num, 2] 
            y: same as x
            the features of axis 1 of x are [azimuth, elevation], respectively.
        return  
            s: array [num]
    ''' 
    x, y = x.T, y.T
    EARTH_REDIUS = 1
    radx, rady = rad(x), rad(y)
    a = radx - rady
    s = 2 * np.arcsin(np.sqrt(np.sin(a[1]/2)**2 + np.cos(radx[1]) * np.cos(rady[1]) * (np.sin(a[0]/2)**2)))
    s = s * EARTH_REDIUS
    # print("distance=",s)
    return s

def testProgam():
    dataPath = ''
    viewMapPath = '/home/zhangxin/View-sensitive-conv/lib/datasets/bin_mid.npy'
    outputPath = '/home/zhangxin/View-sensitive-conv/lib/datasets/pTilde.npy'
    ''' test program '''
    tempViewMap = np.random.randint(-180,180,(100,2))
    generator = GaussianSmoothedProbGenerator(quantizedViewPointPath=viewMapPath, viewMap=tempViewMap)  
    pts = np.random.uniform(-180,180,(10000,2))
    probs = generator.originProbs(pts)
    pTilde = generator.gaussianSmooth(probs)
    generator.weightedAllDataPoint(dataPath)
    print('Done!')


if __name__ == "__main__":
    
    dataPath = ''
    name = '_3'
    noSmoothing = True
    viewMapPath = '/home/zhangxin/View-sensitive-conv/lib/datasets/bin_mid'+name+'.npy'
    outputPath = '/home/zhangxin/View-sensitive-conv/lib/datasets/pTilde'+name+'.npy'
    ''' test program '''
    # testProgam()
    generator = GaussianSmoothedProbGenerator(quantizedViewPointPath=viewMapPath, noSmoothing=noSmoothing)  
    generator.weightedAllDataPoint(outputPath=outputPath)
    print('Done!')
