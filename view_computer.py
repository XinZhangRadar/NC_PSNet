import pdb
def view_label(azimuth,elevation):
    '''
    input: azimuth
           elevation
    out  : view label
    注意分类这里第一项要小于第二项而且第二项不能超过360 注意是个圆
    ''' 
    a_intervals = list(([0,15],[15,75],[75,105],[105,165],[165,195],[195,255],[255,285],[285,345],[345,360]))
    e_intervals = list(([-90,-15],[-15,15],[15,60],[60,90]))
    
    for ix in range(len(e_intervals)) :
        pdb.set_trace()

        max_i = max(e_intervals[ix][0],e_intervals[ix][1])
        min_i = min(e_intervals[ix][0],e_intervals[ix][1])
        if min_i <= elevation and elevation <= max_i:
            e_label = ix
    
    for ix in range(len(a_intervals)) :
        max_i = max(a_intervals[ix][0],a_intervals[ix][1])
        min_i = min(a_intervals[ix][0],a_intervals[ix][1])
        if min_i <= azimuth and azimuth <= max_i:
            a_label = ix%8

    # label calculate function


    try:
        view_label = e_label*(8) + a_label+1
        if view_label > 33 :
            pdb.set_trace()
    except:
        pdb.set_trace()

    return view_label

if __name__ == '__main__':
    azimuth = 6.5
    elevation = 65.5
    print('view label is %d',view_label(azimuth,elevation))
