#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author : Liu Yicheng
# Date : 2019/8/8
import torch
import pytorch_ssim
import numpy as np
import cv2
import os
import pdb
import sklearn 
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torch.autograd import Variable

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from pylmnn import LargeMarginNearestNeighbor as LMNN
from metric_learn import RCA_Supervised

def read_data( path ):
    
    files= os.listdir(path) 
    images = []
    for file in files: #遍历文件夹
        if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
            im = cv2.imread( os.path.join(path,file), cv2.IMREAD_GRAYSCALE )
            # im = cv2.resize(im, (576, 576), interpolation=cv2.INTER_CUBIC)  
            im = np.array(im).reshape(-1)
            if images is None :     
                images.append(im)
            else :
                images.append(im)
    
    return np.array(images),files

def read_data_c( path ):
    #pdb.set_trace()
    files= os.listdir(path) 
    files = dicimg_sort(files)
    images = [] 
    for file in files: #遍历文件夹
        # pdb.set_trace() 
        if not os.path.isdir(file) and file != 'background.jpg': #and file[0:2] == label: #判断是否是文件夹，不是文件夹才打开
            im = cv2.imread( os.path.join(path,file), cv2.IMREAD_GRAYSCALE )
            im = cv2.resize(im, (48, 48), interpolation=cv2.INTER_CUBIC)
            im = np.array(im).reshape(-1) 
            if images is None :     
                images.append(im)
            else :
                images.append(im)
    
    return np.array(images),files

def read_data_new( ):
    
    path1 = r'/home/zhangxin/View-sensitive-conv/img/'
    files= os.listdir(path1) 
    files = img_sort(files)
    #pdb.set_trace()
    images = []
    for file in files: #遍历文件夹
        
        if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
            im = cv2.imread( os.path.join(path1,file), cv2.IMREAD_GRAYSCALE )
            # im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_CUBIC)  
            im = cv2.resize(im, (48, 48), interpolation=cv2.INTER_CUBIC)  
            im = np.array(im).reshape(-1)
            if images is None :     
                images.append(im)
            else :
                images.append(im)
    return np.array(images),files

def img_sort(files):
	#l = ['5.rb', '2.rb', '201.rb', '51.rb', '7.rb', '4.rb']
	#print 'Before:'
	#print l
	for i in range(len(files)):    
		files[i] = files[i].split('.')    
		files[i][0] = int(files[i][0])
	#print 'After:'
	#print l
	files.sort()
	#print 'Sorted:'
	#print l
	for i in range(len(files)):    
		files[i][0] = str(files[i][0])    
		files[i] = files[i][0] + '.' + files[i][1]
	#print 'Recover:'
	#print l
	return files

def dicimg_sort(files):
	#l = ['5.rb', '2.rb', '201.rb', '51.rb', '7.rb', '4.rb']
	#print 'Before:'
	#print l
	for i in range(len(files)):    
		files[i] = files[i].split('.')    
		# files[i][0] = int(files[i][0][0:2])
	#print 'After:'
	#print l
	files.sort()
	#print 'Sorted:'
	#print l
	for i in range(len(files)):    
		files[i][0] = str(files[i][0])    
		files[i] = files[i][0] + '.' + files[i][1]
	#print 'Recover:'
	#print l
	return files

def class_name(files):
    for i in range(len(files)):    
        files[i] = files[i].split('.')    
    for i in range(len(files)):    
        if(files[i][0][2] == '-'):
            files[i] = (int(files[i][0][3]) - 1) * 8 + int(files[i][0][5])
        elif(files[i][0][1] == '-'):
            files[i] = (int(files[i][0][2]) - 1) * 8 + int(files[i][0][4])
        else:
            del files[i]
    return files


def Img_PyLMNN(X_train, y_train, X_test, y_test,files_img):
    # # Load a data set
    # X, y = load_iris(return_X_y=True)

    # # Split in training and testing set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

    # # Set up the hyperparameters
    X = X_train
    Y = y_train
    #X_train = X[192:264]
    #y_train = Y[192:264]
    #X_test = X[200:]
    #y_test = Y[200:]
    
    k_train, k_test, n_components, max_iter = 20, 20, 120, 120
    pdb.set_trace()
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    # # Instantiate the metric learner
    lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter, verbose=1)

    # Train the metric learner
    lmnn.fit(X_train, y_train)

    # Fit the nearest neighbors classifier
    knn = KNeighborsClassifier(n_neighbors=k_test)
    knn.fit(lmnn.transform(X_train), y_train)



    # Compute the k-nearest neighbor test accuracy after applying the learned transformation
    lmnn_acc = knn.score(lmnn.transform(X_test), y_test)
    print('knn_result')
    print(knn.predict(lmnn.transform(X_test)))
    print('y_test')
    print(y_test)
    print('knn_score')
    print(lmnn_acc)
    # print('LMNN accuracy on test set of {} points: {:.4f}'.format(X_test.shape[0], lmnn_acc))
    return X_test.shape[0], lmnn_acc, lmnn, knn


def Img_RCA(X_train, y_train, X_test, y_test):
    # # Load a data set
    # X, y = load_iris(return_X_y=True)

    # # Split in training and testing set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)

    # # Set up the hyperparameters
        # # Set up the hyperparameters
    X = X_train
    Y = y_train
    # X_train = X[0:200]
    # y_train = Y[0:200]
    # X_test = X[200:]
    # y_test = Y[200:]
    X_train = X
    y_train = Y
    X_test = X
    y_test = Y
    k_train, k_test, n_components, max_iter = 10, 10, 10, 120
    
    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)



    ## Instantiate the metric learner
    # pdb.set_trace()
    RCA = RCA_Supervised(num_chunks=20, chunk_size=5)

    # # Train the metric learner
    RCA.fit(X_train, y_train)
    # from metric_learn import NCA
    # from sklearn.datasets import make_classification
    # from sklearn.neighbors import KNeighborsClassifier
    # nca = NCA()

    # nca.fit(X_train, y_train)

    # Fit the nearest neighbors classifier
    # knn = KNeighborsClassifier(metric=nca.get_metric())
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(RCA.transform(X_train), y_train)
    # knn.fit(X_train, y_train)

    # Compute the k-nearest neighbor test accuracy after applying the learned transformation
    RCA_acc = knn.score(RCA.transform(X_test), y_test)
    # NCA_acc = knn.score(X_test, y_test)



    print('knn_result')
    print(knn.predict(X_test))
    print('y_test')
    print(y_test)
    print('knn_score')
    print(RCA_acc)
    import pdb; pdb.set_trace()
    # print('LMNN accuracy on test set of {} points: {:.4f}'.format(X_test.shape[0], lmnn_acc))
    return X_test.shape[0], NCA_acc, RCA, knn


if __name__ == "__main__":

    path =  r"/home/zhangxin/View-sensitive-conv/dictimage/" 
    images_dic, files_dic = read_data_c(path)
    # new_image, _ = read_data_new()
    image_test, files_img = read_data_new()
    y_train = np.expand_dims(np.array(class_name(files_dic)), axis = 1)
    y_train = y_train.reshape(-1)
    for i in range(len(files_img)):    
        files_img[i] = files_img[i].split('.')    
    for i in range(len(files_img)):    
        files_img[i] = int(files_img[i][0])
    # y_test = np.expand_dims(np.array(files_img), axis = 1)
    #y_test = np.array(files_img)
    y_test = np.array([4,5,14,14,22,22,12,12,12,20])
    test, LMNN_acc,lmnn,knn = Img_PyLMNN(images_dic, y_train, image_test, y_test,files_dic)
    #test, rca_acc,RCA,knn = Img_RCA(images_dic, y_train, image_test, y_test)
    pdb.set_trace()
    # new_image = np.array(new_image).reshape(10, 256, 256)
    # images_db = np.array(images_db).reshape(24, 256, 256)
    # new_image = np.expand_dims(new_image,axis=0)
    # images_db = np.expand_dims(images_db,axis=0)

    # # SSIM method for distance
    # ssim_out = []
    # ssim_out1 = []
    # SSim = []
    # SSim_loss = []
    # for i in range(images_db.shape[1]):
    #     for j in range(new_image.shape[1]):
    #         img1 = Variable(torch.from_numpy(np.expand_dims(images_db[:,i,:,:],axis=0))).type(torch.FloatTensor)
    #         img2 = Variable(torch.from_numpy(np.expand_dims(new_image[:,j,:,:],axis=0))).type(torch.FloatTensor)
    #         if torch.cuda.is_available():
    #             img1 = img1.cuda()
    #             img2 = img2.cuda()
    #         #pdb.set_trace()
    #         #SSim.append(pytorch_ssim.ssim(img1, img2))
    #         # if torch.cuda.is_available():
    #         #     img1 = img1.cuda()
    #         #     img2 = img2.cuda()
    #         # SSim.append(pytorch_ssim.ssim(img1, img2))
    #         if(i == 0 and j == 0):
    #             SSim = pytorch_ssim.ssim(img1, img2).unsqueeze(0)
    #         else:
    #             SSim1 = pytorch_ssim.ssim(img1, img2).unsqueeze(0)
    #             SSim = torch.cat([SSim, SSim1], 0)
            
            # print(pytorch_ssim.ssim(img1, img2))
            # ssim_loss = pytorch_ssim.SSIM(window_size = 11)
            # ssim_out = ssim_loss(img1, img2)
            # pdb.set_trace()

            # if(i == 0 and j == 0):
            #     ssim_loss = pytorch_ssim.SSIM(window_size = 11)
            #     ssim_out = ssim_loss(img1, img2).unsqueeze(0)
            #     # pdb.set_trace()
            # else:
            #     ssim_loss1 = pytorch_ssim.SSIM(window_size = 11)
            #     ssim_out1 = ssim_loss1(img1, img2).unsqueeze(0)
            #     ssim_out = torch.cat([ssim_out, ssim_out1], 0)
            #ssim_loss = pytorch_ssim.SSIM(window_size = 11)
            #torch.cat([a, b], 0)
            #SSim_loss ssim_loss(img1, img2))
            # pdb.set_trace()
    # SSim = SSim.reshape(24, 10)
    # pdb.set_trace()
    # """PCA"""
    # pca = PCA(70)
    # images = pca.fit_transform(images_db)

    # pca1 = PCA(70)
    # c = pca1.fit_transform(new_image)
    # pdb.set_trace()    


    # """Gausian Mixture Model"""
    # # gmm = GaussianMixture(n_components= 24, verbose=1).fit(images)
    # # gmm = gmm.fit(images)
    # # labels = gmm.predict(images)

    # gmm = GaussianMixture(n_components= 24, verbose=1).fit(images)
    # gmm = gmm.fit(images)
    # labels = gmm.predict(images) 
    # pre_labels = gmm.predict(new_image)
    
    # """visualizing data label"""
    # pca_im = PCA(n_components=3)
    # image_vis = pca_im.fit_transform(images)
    # visual = np.split(image_vis , 3, axis= 1)

    # #mean
    # pca_vis = PCA(n_components= 3)
    # visual_mean = pca_vis.fit_transform(gmm.means_)
    # visual_mean = np.split(visual_mean , 3, axis= 1)

    # # for file,lab in zip(files,labels):
    # #     # if lab == 2:
    # #         print('%s:%d'%(file,lab))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(visual_mean[1].squeeze(),visual_mean[0].squeeze(), visual_mean[2].squeeze(), c='y')
    # j = 0
    # for x, y, z in zip(visual_mean[1].squeeze(),visual_mean[0].squeeze(), visual_mean[2].squeeze()):
        
    #     ax.text(  x, y, z, str(j))
    #     j += 1

    # ax.scatter(visual[1].squeeze(),visual[0].squeeze(), visual[2].squeeze(), c='r')

    # x = visual[1].squeeze()
    # y = visual[0].squeeze()
    # z = visual[2].squeeze()
    # for i in range(len(labels)):
    #     ax.text(  x[i], y[i], z[i], str(labels[i]))
    
    # plt.show()

    # print("Done!!")



