import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')
import os
#import sys
import zipfile, rarfile
import struct
import pandas as pd
import numpy as np
import tables as tb
import time
import pdb

def getZ(filename):
    name, end = os.path.splitext(filename)
    if end == '.rar':
        Z = rarfile.RarFile(filename)
    elif end == '.zip':
        Z = zipfile.ZipFile(filename)
    return Z

class Bunch(dict):

    def __init__(self, *args, **kwds):
        super(Bunch,self).__init__(*args, **kwds)
        self.__dict__ = self

class MPF:

    def __init__(self, fp):
        self.fp = fp
        #pdb.set_trace()
        header_size = struct.unpack('i', self.fp.read(4))[0]
        self.code_format = self.fp.read(8).decode('ascii').rstrip('\x00')
        self.text = self.fp.read(header_size - 62).decode().rstrip('\x00')
        self.code_type = self.fp.read(20).decode('ascii').rstrip('\x00')
        self.code_length = struct.unpack('h', self.fp.read(2))[0]
        self.data_type = self.fp.read(20).decode('ascii').rstrip('\x00')
        self.nrows = struct.unpack('i', self.fp.read(4))[0]
        self.ndims = struct.unpack('i', self.fp.read(4))[0]

    def __iter__(self):
        m = self.code_length + self.ndims 
        
        for i in range(0, m * self.nrows, m):
            label = self.fp.read(self.code_length).decode('gbk')
            data = np.frombuffer(self.fp.read(self.ndims), np.uint8)
            yield data, label

class MPFBunch(Bunch):

    def __init__(self, root, set_name, *args, **kwds):
        super(MPFBunch,self).__init__(*args, **kwds)
        filename, end = os.path.splitext(set_name)
        

        if 'HW' in filename and end == '.zip':
            if '_' not in filename:
                self.name = filename
                Z = getZ(root+set_name)
                self._get_dataset(Z)
        else:
            
            pass

    def _get_dataset(self, Z):
        
        for name in Z.namelist():
            if name.endswith('.mpf'):
                writer_ = 'writer'+os.path.splitext(name)[0].split('/')[1]
                

                with Z.open(name,'U') as fp:
                    mpf = MPF(fp)
                    self.text = mpf.text
                    self.nrows = mpf.nrows
                    self.ndims = mpf.ndims
                    #pdb.set_trace()
                    #bb = [label for label in iter(mpf)]
                    #aa = {label : data for data, label in iter(mpf)}
                    #pdb.set_trace()
                    db = Bunch({label : data for data, label in iter(mpf)})
                    self[writer_] = pd.DataFrame.from_dict(db).T

class BunchHDF5(Bunch):
    '''

    '''
    def __init__(self, mpf, *args, **kwds):
        super(Bunch,self).__init__(*args, **kwds)

        if 'name' in mpf:
            print(mpf.name+'reading statusbars:')

        start = time.time()  
        for i, wname in enumerate(mpf.keys()):
            if wname.startswith('writer'):
                _dir = root+'mpf/'
                if not os.path.exists(_dir):
                    os.mkdir(_dir)
                self.path = _dir+mpf.name+'.h5'
                mpf[wname].to_hdf(self.path, key = wname, complevel = 7)
                k = sys.getsizeof(mpf)     # mpf 
                print('-' * (1 + int((time.time() - start) / k)))

            if i == len(mpf.keys()) - 1:
                print('\n')

class XCASIA(Bunch):

    def __init__(self, root, *args, **kwds):
        super(XCASIA,self).__init__(*args, **kwds)
        #pdb.set_trace();
        self.paths = []
        print('write in disk')
        start = time.time()
        for filename in os.listdir(root):
            self.mpf = MPFBunch(root, filename)
            BunchHDF5(self.mpf)
        print('Time'+str(time.time() - start)+'second.')



if __name__ == '__main__':
    root = '/home/zhangxin/CASIA_data/'
    #pdb.set_trace();
    xa = XCASIA(root);
    mpf_root = root+'mpf/'
 

    #for filename in os.listdir(mpf_root):
    #    h5 = tb.open_file(mpf_root+filename)
    #    break
    #df = pd.read_hdf(mpf_root+filename, key='writer1001')
    #print(df)
