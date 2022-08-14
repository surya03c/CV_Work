import numpy as np
from matplotlib import image as im

def isInvertible(arr):
    return (np.linalg.det(np.matmul(np.transpose(arr),arr)) != 0)


def isValidEig(arr, bound):
    vals,vecs = np.linalg.eig(np.matmul(np.transpose(arr),arr))
    eig1 = vals[0]
    eig2 = vals[1]

    if (eig1 > bound and eig2 > bound):
        if (eig1 > eig2):
            if (abs(eig2/eig1) < 0.5) return True
        else if (eig1 < eig2):
            if (abs(eig1/eig2) < 0.5) return True
    return False
     


def lucas_kanade(image1, image2, row, col, window_size=20):
    '''
    Returns optical flow gradient vector tuple (u,v) after accepting 2 image inputs, frame rate, init row/col, and desired window size for implementation 
    of Lucas-Kanade Optical Flow algorithm  
    '''

    #assuming we are dealing with 8 bit images but still want to normalize
    im1 = im.imread(image1)/255 
    im2 = im.imread(image2)/255

    assert(im1.size == im2.size)

    arr = np.zeros((400,2))
    out = np.zeros((400,1))
    r_track = 0
    c_track = 0
    out_track = 0
    i = 0
    j = 0 

    #fills in the necessary arrays
    while (i < window_size):
        while (j < window_size):
            arr[r_track][c_track] = (0.25*(im1[row+i][col+j+1] + im2[row+i][col+j+1] + im1[row+i+1][col+j+1] + im2[row+i+1][col+j+1])) - (0.25*(im1[row+i][col+j] + im2[row+i][col+j] + im1[row+i+1][col+j] + im2[row+i+1][col+j]))
            c_track++
            arr[r_track][c_track]= (0.25*(im1[row+i+1][col+j] + im2[row+i+1][col+j] + im1[row+i+1][col+j+1] + im2[row+i+1][col+j+1])) - (0.25*(im1[row+i][col+j] + im2[row+i][col+j] + im1[row+i][col+j+1] + im2[row+i][col+j+1]))
            r_track++
            out[out_track] = (0.25*(im2[row+i][col+j] + im2[row+i+1][col+j] + im2[row+i+1][col+j+1] + im2[row+i][col+j+1])) - (0.25*(im1[row+i][col+j] + im1[row+i+1][col+j] + im1[row+i+1][col+j+1] + im1[row+i][col+j+1]))
            j++
        i++
        j = 0

    

    assert(isInvertible(arr))
    assert(isValidEig(arr, bound)) #FILL IN A BOUND VAL

    return np.matmul((np.matmul(np.linalg.inv(np.matmul(np.transpose(arr),arr)), np.transpose(arr))), out)









