from __future__ import division
from Dataset import *

from math_utils import *

class P_estimation(MathUtils):
   
    def __init__(self):
        MathUtils.__init__(self)

    def get_classconditional_probability(self, x, dataset, h_result):

        index_h =h_result
        cc = 1
        for index_name in dataset.indexnamelist:
            cc *= index_h[index_name]
        coefficient = 1.0 / (dataset.num * cc)
        pp=0
        for i in range(dataset.num):
            p=1
            for index_name in dataset.data.keys():
              
                x_i=dataset.data[index_name].iloc[i]

                XX=float(x[index_name-dataset.data.keys()[0]]-x_i)/float(index_h[index_name])
                p_kerel=self.kernel_function(XX,"Gauss")
                p*=p_kerel
        
            pp+=p
        probility=coefficient*pp
   
        return probility

    def get_h(self,dataset):
        index_h={}
        coefficient = 1
        cc=1
        for index_name in dataset.indexnamelist:
            index_max=max(dataset.data[index_name])
            index_min=min(dataset.data[index_name])
            index_h[index_name]=(index_max-index_min)/float(3000**0.5)
        
        return index_h,coefficient