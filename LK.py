from curses import window
import numpy as np
from matplotlib import image as im
from matplotlib import pyplot as plt
from scipy import signal as sig
import cv2 as cv
import random as r

def isInvertible(arr):
    return (np.linalg.det(np.matmul(np.transpose(arr),arr)) != 0)


def isValidEig(arr, bound):
    vals,vecs = np.linalg.eig(np.matmul(np.transpose(arr),arr))
    eig1 = vals[0]
    eig2 = vals[1]
    if (eig1 > bound and eig2 > bound):
        if (eig1 > eig2):
            return (abs(eig2/eig1) > 0.1)
        elif (eig1 < eig2):
            return (abs(eig1/eig2) > 0.1)
    return False

def optimalFeatureSelection(image1, window_size=5):
    dst = cv.cornerHarris(np.float32(cv.cvtColor(cv.imread(image1),cv.COLOR_BGR2GRAY)),2,3,0.01)
    corners = []
    i = window_size
    j = window_size
    while (i < dst.shape[0]- window_size):
        while (j<dst.shape[1]-window_size):
            if (dst[i][j] > 0.01*dst.max()):
                corners.append((i,j))
            j+=1
        i+=1
        j=0
    
    ret = []
    track = 0

    #sampling of 20 random feature vectors to show 
    while (track < 20):
        ret.append(r.choice(corners))
        track+=1;

    return ret;

def lucas_kanade(image1, image2, xindex, yindex, window, eig_lim = 0.01):
    '''
    Returns optical flow gradient vector tuple (u,v) after accepting 2 image inputs, init row/col, minimum eigenvalue limit, and desired window size for implementation 
    of Lucas-Kanade Optical Flow algorithm  
    '''
    im1 = im.imread(image1) 
    im2 = im.imread(image2)

    #grayscaling process

    R, G, B = im1[:,:,0], im1[:,:,1], im1[:,:,2]
    im1 = 0.2989 * R + 0.5870 * G + 0.1140 * B
    r, g, b = im2[:,:,0], im2[:,:,1], im2[:,:,2]
    im2 = 0.2989 * r + 0.5870 * g + 0.1140 * b

    assert(im1.size == im2.size)

    # will use sobel kernels
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    time_kernel = np.array([[1,1,1], [1,1,1], [1,1,1]])

    F_x = sig.convolve2d(im1, x_kernel, boundary = 'symmetric', mode='same')
    F_y = sig.convolve2d(im1, y_kernel, boundary = 'symmetric', mode = 'same')
    F_t = sig.convolve2d(im2, time_kernel, boundary = 'symmetric', mode = 'same') - sig.convolve2d(im1, -time_kernel, boundary = 'symmetric', mode = 'same')
    

    #Starting from a point (row,col) and given a window size, I want to iterate through that window and fill in the gradients that are needed. From there I can 
    #proceed with matrix calculations and so on

    Ix = np.zeros((window,window))
    Iy = np.zeros((window,window))
    It = np.zeros((window,window))
    r = 0
    c = 0

    for i in range(yindex, yindex+window):
        for j in range(xindex, xindex+window):
            Ix[r,c] = F_x[i,j]
            Iy[r,c] = F_y[i,j]
            It[r,c] = F_t[i,j]
            c+=1
        r+=1
        c=0
    
    A = np.column_stack((Ix.flatten(),Iy.flatten()))
    B = It.flatten()
    if (isInvertible(A) and isValidEig(A,eig_lim)):
        return np.matmul((np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)), np.transpose(A))), B)
    
    # arr = np.zeros(im1.shape)
    # out = np.zeros()
    # r_track = 0
    # c_track = 0
    # out_track = 0
    # i = 0
    # j = 0 

    # #fills in the necessary arrays
    # #8/15: will replace this with sobel gradient implementation from opencv package and see if there is any difference
    # while (i < window_size):
    #     while (j < window_size):
    #         arr[r_track][c_track] = (0.25*(im1[row+i][col+j+1] + im2[row+i][col+j+1] + im1[row+i+1][col+j+1] + im2[row+i+1][col+j+1])) - (0.25*(im1[row+i][col+j] + im2[row+i][col+j] + im1[row+i+1][col+j] + im2[row+i+1][col+j]))
    #         c_track+=1
    #         arr[r_track][c_track]= (0.25*(im1[row+i+1][col+j] + im2[row+i+1][col+j] + im1[row+i+1][col+j+1] + im2[row+i+1][col+j+1])) - (0.25*(im1[row+i][col+j] + im2[row+i][col+j] + im1[row+i][col+j+1] + im2[row+i][col+j+1]))
    #         r_track+=1
    #         c_track=0
    #         out[out_track] = (0.25*(im2[row+i][col+j] + im2[row+i+1][col+j] + im2[row+i+1][col+j+1] + im2[row+i][col+j+1])) - (0.25*(im1[row+i][col+j] + im1[row+i+1][col+j] + im1[row+i+1][col+j+1] + im1[row+i][col+j+1]))
    #         out_track+=1
    #         j+=1
    #     i+=1
    #     j = 0

    

    #assert(isInvertible(arr))
    #assert(isValidEig(arr, eig_lim)) #FILL IN A BOUND VAL

    # print("compiled successfully")

    # return np.matmul((np.matmul(np.linalg.inv(np.matmul(np.transpose(arr),arr)), np.transpose(arr))), out)


def main():
    #An example run of LK in action
    xindex,yindex = (46,52) # manually inputted coordinates of a textured object (person) moving at a reasonable speed
    u,v = lucas_kanade("test1.jpeg", "test2.jpeg", xindex,yindex, window=10) # window is small to reflect person's size in image, also to limit impact from non-textured regions
    plt.imshow(im.imread("test1.jpeg"))
    plt.quiver(xindex,yindex, u,v ,color='r')
    plt.show()

if __name__ == "__main__":
    main()











