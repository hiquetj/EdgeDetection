import numpy as np
import matplotlib
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy import signal
from scipy import misc

def sobel_filter(im, kernelSize):
    #make image a numpy array to be able to do convolution 
    im = im.astype(np.float)
    width, height = im.shape
    img = im
     
    #just to make sure kernelsize is something we expect 
    assert(kernelSize == 3 or kernelSize == 5);
     
    #here is were we compute with the horizontal and vertical derivative approximations 
    #either 5x5 or 3x3
    if kernelSize == 3:
        k_horiz = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
        k_vert = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)
    else:
        k_horiz = np.array([[-1, -2, 0, 2, 1], 
                   [-4, -8, 0, 8, 4], 
                   [-6, -12, 0, 12, 6],
                   [-4, -8, 0, 8, 4],
                   [-1, -2, 0, 2, 1]], dtype = np.float)
        k_vert = np.array([[1, 4, 6, 4, 1], 
                   [2, 8, 12, 8, 2],
                   [0, 0, 0, 0, 0], 
                   [-2, -8, -12, -8, -2],
                   [-1, -4, -6, -4, -1]], dtype = np.float)
    
    #convolution is easy in python just call this function  
    gx = signal.convolve2d(img, k_horiz, mode='same', boundary = 'symm', fillvalue=0)
    gy = signal.convolve2d(img, k_vert, mode='same', boundary = 'symm', fillvalue=0)
 	
 	#do the next computation to top it all off
    g = np.sqrt(gx * gx + gy * gy)
    g *= 255.0 / np.max(g)
    
    #this is for the peak threshold detection. I chose 50, could chose something higher if need be
    g[g<50] = 0
    			   
    return g
 
#luckily python lets us choose a jpg    
img = ndimage.imread('greyscale_input.jpg')
#do a 3x3 filter on it
x = sobel_filter(img, 3)
misc.imsave('edge3x3_PEAK_out.jpg', x)
#do a 5x5 filter on it too
y = sobel_filter(img, 5)
misc.imsave('edge5x5_PEAK_out.jpg', y)
