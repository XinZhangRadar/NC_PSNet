import numpy as np
import cv2 as cv
import pdb



def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
    Ai - is an affine transform matrix from skew_img to img
    
    '''
    #pdb.set_trace();
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        #img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    #Ai = cv.invertAffineTransform(A)
    return img, mask, A#, Ai,

'''
if __name__ == '__main__':
    img = cv.imread('1.jpg');
     
    params = [(1.0, 0.0)]
    l = 0;
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            #params.append((t, phi))
            timg, tmask, Ai = affine_skew(t, phi, img)
            l+=1;
            #name = 'pic' + '_t'+str(t)+'_'+'phi'+str(phi)+'.jpg'
            name = 'pic_test'+str(l)+'.jpg'
            cv.imwrite(name,timg);


def transform(A,X):

    XX = np.row_stack((X,np.ones((1,X.shape[1]))));
    X_vector = np.dot(A[0,:],XX);
    Y_vector = np.dot(A[1,:],XX);
    xmin = np.min(X_vector);
    xmax = np.max(X_vector);
    ymin = np.min(Y_vector);
    ymax = np.max(Y_vector);
    return xmin,xmax,ymin,ymax
'''