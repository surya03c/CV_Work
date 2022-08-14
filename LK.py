import numpy as np
from matplotlib import image as im

def isInvertible(arr):
    return (np.linalg.det(np.matmul(np.transpose(arr),arr)) != 0)


def isValidEig(arr, bound):
    vals,vecs = np.linalg.eig(np.matmul(np.transpose(arr),arr))
    eig1 = vals[0]
    eig2 = vals[1]
    print(str(eig1))
    print(str(eig2))

    if (eig1 > bound and eig2 > bound):
        if (eig1 > eig2):
            return (abs(eig2/eig1) > 0.5)
        elif (eig1 < eig2):
            return (abs(eig1/eig2) > 0.5)
    return False
     


def lucas_kanade(image1, image2, row, col, eig_lim, window_size=20):
    '''
    Returns optical flow gradient vector tuple (u,v) after accepting 2 image inputs, init row/col, minimum eigenvalue limit, and desired window size for implementation 
    of Lucas-Kanade Optical Flow algorithm  
    '''

    #assuming we are dealing with 8 bit images but still want to normalize
    im1 = im.imread(image1)/255 
    im2 = im.imread(image2)/255

    R, G, B = im1[:,:,0], im1[:,:,1], im1[:,:,2]
    im1 = 0.2989 * R + 0.5870 * G + 0.1140 * B
    r, g, b = im2[:,:,0], im2[:,:,1], im2[:,:,2]
    im2 = 0.2989 * r + 0.5870 * g + 0.1140 * b

    print(im1.shape)

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
            c_track+=1
            arr[r_track][c_track]= (0.25*(im1[row+i+1][col+j] + im2[row+i+1][col+j] + im1[row+i+1][col+j+1] + im2[row+i+1][col+j+1])) - (0.25*(im1[row+i][col+j] + im2[row+i][col+j] + im1[row+i][col+j+1] + im2[row+i][col+j+1]))
            r_track+=1
            c_track=0
            out[out_track] = (0.25*(im2[row+i][col+j] + im2[row+i+1][col+j] + im2[row+i+1][col+j+1] + im2[row+i][col+j+1])) - (0.25*(im1[row+i][col+j] + im1[row+i+1][col+j] + im1[row+i+1][col+j+1] + im1[row+i][col+j+1]))
            out_track+=1
            j+=1
        i+=1
        j = 0

    

    assert(isInvertible(arr))
    assert(isValidEig(arr, eig_lim)) #FILL IN A BOUND VAL

    print("compiled successfully")

    return np.matmul((np.matmul(np.linalg.inv(np.matmul(np.transpose(arr),arr)), np.transpose(arr))), out)


def main():
    print(str(lucas_kanade("I1.jpeg", "I2.jpeg", 75, 75, 0.1)))

if __name__ == "__main__":
    main()











