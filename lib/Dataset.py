from math_utils import MathUtils
import pandas as pd
import copy
from sklearn.cross_validation import *


class Dataset(MathUtils):
    def __init__(self):
        MathUtils.__init__(self)
        self.name = None
        self.data = None
        self.classnamelist = [] 
        self.indexnamelist = [] 
        self.classnum = None    
        self.indexnum = None 
        self.num = None       
        self._data_by_class ={}
        self.prior_probability = {}


    def importdata(self,**kwargs):
        if "filename" in kwargs:
            fname=kwargs["filename"]
            data = self.input_data_by_csv(fname, kwargs["index_col"] )
            self.data = data[0]
            self.indexnum = data[3]
            self.num = data[4]
            self.indexnamelist = data[2]
            self.classnamelist = data[1]
            self.class_list=data[5]
            return None


        elif "framedata" in kwargs:
            self.data  = kwargs["framedata"]
            self.classnamelist=list(set(self.data.axes[0]))
            self.indexnamelist=list(set(self.data.keys()))
            self.indexnum=len(self.indexnamelist)
            self.classnum=len(self.classnamelist)
            self.num=len(self.data.axes[0])
            return None

        elif "url" in kwargs:
            dataset = pd.read_csv(kwargs["url"],index_col=kwargs["index_col"],header=None, usecols= kwargs["col_ind"])
            self.importdata(framedata=dataset)
            self.indexnamelist = kwargs["index_name"]
            return None

        elif "data_dic" in kwargs:
            return None


    def get_group(self):
        for i_index in self.classnamelist:
            self._data_by_class[i_index] = self.data.xs(i_index)
            self.prior_probability[i_index] = 1.0/ float(len(self.classnamelist)) 

class TrainData(Dataset):

    def __init__(self):
        Dataset.__init__(self)
        self.covMatrix = None
        self.mean = None
        self.classCovMatrix = {}
        self.classMean = {}
        self.folds_data = []

    def split_into_folds(self, n_folds):
        kf = KFold(self.num, n_folds, shuffle = False)
        for iteration, index_data in (kf):
            a = self.data.iloc[iteration]
            b = self.data.iloc[index_data]
            a1 =copy.deepcopy(a)
            b1 =copy.deepcopy(b)

            train_data_i = TrainData()
            train_data_i.importdata(framedata = a1)

            train_data_i.indexnamelist = self.indexnamelist
            test_data_i = TestData()
            train_data_i.name = self.name
            test_data_i.name =self.name
            test_data_i.importdata(framedata = b1)
            test_data_i.indexnamelist = self.indexnamelist
            self.folds_data.append([train_data_i, test_data_i])

class TestData(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.result = []
        self.target_list = []

    def importdata(self,**kwargs):
        self.data = kwargs["framedata"]
        self.classnamelist = list(set(self.data.axes[0]))
        self.indexnamelist = list(set(self.data.keys()))
        self.indexnum = len(self.indexnamelist)
        self.classnum = len(self.classnamelist)
        self.num = len(self.data.axes[0])
        self.target_list = list(self.data.index)
