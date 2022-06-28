import numpy as np
import pdb 
class BIN_MID():
    def __init__(self, bin_num = 32):
        self.bin_num = bin_num
        if bin_num == 32 :
            self.bin_mid = self.compute_bin_mid_32(np.zeros([self.bin_num, 2]))
        elif bin_num == 24:
            self.bin_mid = self.compute_bin_mid_24(np.zeros([self.bin_num, 2]))
        elif bin_num == 12:
            self.bin_mid = self.compute_bin_mid_12(np.zeros([self.bin_num, 2]))
        elif bin_num == 3:
            self.bin_mid = self.compute_bin_mid_3(np.zeros([self.bin_num, 2]))
    def compute_bin_mid_3(self,binmid):

        #a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal
        e_intervals = list(([-90,-45],[-45,45],[45,90]))

        v_id = 0
        for jx in range(len(e_intervals)) :
            
            mid_e = (e_intervals[jx][0]+e_intervals[jx][1])/2
            binmid[v_id,:] = 0,mid_e
            v_id = v_id + 1
        return binmid

    def compute_bin_mid_12(self,binmid):

        a_intervals = list(([0,45],[45,135],[135,225],[225,315],[315,360]))# pascal
        e_intervals = list(([-90,-45],[-45,45],[45,90]))


        v_id = 0
        for jx in range(len(e_intervals)) :
            
            mid_e = (e_intervals[jx][0]+e_intervals[jx][1])/2
            
            for ix in range(len(a_intervals)-1) :
                if ix == 0 :
                    mid_a = 0
                else :
                    mid_a = (a_intervals[ix][0]+a_intervals[ix][1])/2
                binmid[v_id,:] = mid_a,mid_e
                v_id = v_id + 1
        return binmid





    def compute_bin_mid_24(self,binmid):

        a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal
        e_intervals = list(([-90,-45],[-45,45],[45,90]))


        v_id = 0
        for jx in range(len(e_intervals)) :
            
            mid_e = (e_intervals[jx][0]+e_intervals[jx][1])/2
            
            for ix in range(len(a_intervals)-1) :
                if ix == 0 :
                    mid_a = 0
                else :
                    mid_a = (a_intervals[ix][0]+a_intervals[ix][1])/2
                binmid[v_id,:] = mid_a,mid_e
                v_id = v_id + 1
        return binmid        

    def compute_bin_mid_32(self,binmid):

        a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))# pascal
        e_intervals = list(([-90,-15],[-15,15],[15,60],[60,90]))


        v_id = 0
        for jx in range(len(e_intervals)) :
            
            mid_e = (e_intervals[jx][0]+e_intervals[jx][1])/2
            
            for ix in range(len(a_intervals)-1) :
                if ix == 0 :
                    mid_a = 0
                else :
                    mid_a = (a_intervals[ix][0]+a_intervals[ix][1])/2
                binmid[v_id,:] = mid_a,mid_e
                v_id = v_id + 1
        return binmid

    def generate_npy(self,bin_num):
        np.save('bin_mid_{}.npy'.format(bin_num),self.bin_mid)
        return





if __name__ == '__main__':
    pdb.set_trace()
    Bin = BIN_MID(3)
    Bin.generate_npy(3)


