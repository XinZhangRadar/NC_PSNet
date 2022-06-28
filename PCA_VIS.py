import numpy as np
from sklearn.decomposition import PCA
import pickle 
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import softmax
data_name = 'aa'
# pkl_path = 'C:\\Users\\dell\\Desktop\\data\\frcnn\\vis1\\'
pkl_path = '/home/zhangxin/View-sensitive-conv/pkl/'


classes = ('__background__','aeroplane', 'motorbike', 'bench', 'bicycle', 'boat', 'car','diningtable','faucet','desk_lamp','helmet')
class_to_ind = dict(zip(classes, range(classes.__len__())))

file_list = os.listdir(pkl_path)
fmaps = []
classname = []
group = []
check = '2009_004175'
for i,pkl in enumerate(file_list):
    # if pkl.split('.')[-1] == 'pkl' and check == pkl.split('.')[0] :
    if pkl.split('.')[-1] == 'pkl' :
        with open(os.path.join(pkl_path,pkl),'rb') as file :
            [fmaps_p, classname_p] = pickle.load(file)
            for i in range(len(classname_p)):
                if classname_p[i] == '__background__':
                    classname_p[i] = 'background'
            #import pdb;pdb.set_trace()
            ##
            # if  pkl.split('.')[0] == '2009_004175':
            # import pdb; pdb.set_trace()
            # a = fmaps_p.max(axis=1)/fmaps_p.max(axis=1).mean()
            # a = fmaps_p.max(axis=0).mean()/fmaps_p.max(axis=0)
            
            fmaps_p = softmax(fmaps_p)
            # fmaps_p = a.reshape(-1)*fmaps_p
            # fmaps_p = np.exp(fmaps_p)/np.exp(fmaps_p).sum(axis=1).reshape(-1,1) 
            # import pdb; pdb.set_trace()
            # fmaps_p = normalize(fmaps_p,norm='max',axis=0)
            # fmaps_p = (fmaps_p-fmaps_p.mean())/(fmaps_p.max()-fmaps_p.min())
            fmaps.extend(fmaps_p)
            group.extend([str(pkl.split('.')[0])]*len(fmaps_p) )
            classname.extend(classname_p)
# import pdb; pdb.set_trace()
fmaps = np.asarray(fmaps)

num_object = len(fmaps)
max_len = max(np.size(l) for l in fmaps)
matrix = np.zeros((num_object,max_len))

for i, fmap in enumerate(fmaps):
    matrix[i][:np.size(fmap)] = fmap

pca_vis = PCA(2)
cord = pca_vis.fit_transform(matrix).T


sns.set()

ax = sns.scatterplot(x=cord[0], y=cord[1], 
                     hue=classname, 
                    #  size=classname,
                    #  sizes=(200,300), 
                     style=group,
                     legend='full',
                     alpha=0.9)
#plt.show()

plt.savefig('pca.jpg')