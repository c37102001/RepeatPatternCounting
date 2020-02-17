import numpy as np
import math
from numpy.lib.stride_tricks import as_strided
from ipdb import set_trace as pdb


def get_size(w, ks, s=1, p=0):
    '''
    Args:
        w: 1D image size (either height or width),
        ks: 1D window size (height or width corresponding to w)
        s: stride of sliding window
        p: padding
    '''
    return math.floor((w + 2*p - ks) / s) + 1


def sliding_window_3D(img, window_size, strd_size):
    '''
    Args:
        img (ndarray): colored image sized[3, h, w], channel must in first dimension.
        window_size (int or tuple): Size of the sliding window
        strd_size (int or tuple): Size of the strides of sliding window
    Returns:
        patches (ndarray): sized[#windows, 3(channels), window_height, window_width]
    '''

    if isinstance(window_size, int):
        h_window, w_window = window_size, window_size
    else:
        h_window, w_window = window_size[0], window_size[1]
    
    if isinstance(strd_size, int):
        h_strd, w_strd = strd_size, strd_size
    else:
        h_strd, w_strd = strd_size[0], strd_size[1]

    c, h, w = img.shape

    new_shape = (c, get_size(h, h_window, h_strd), get_size(w, w_window, w_strd), h_window, w_window)
    patches = as_strided(img, shape=new_shape, strides=(w*h, w*h_strd, 1*w_strd, w, 1))  # sized[C,#,#,2,3]
    patches = patches.reshape(c, -1, h_window, w_window) # sized[C,#,2,3]
    patches = patches.swapaxes(0,1)    # sized[#,C,2,3]
    
    return patches


def sliding_window_2D(img, window_size, strd_size):
    '''
    Args:
        img (ndarray): gray-scale image sized[h, w]
        window_size (int or tuple): Size of the sliding window
        strd_size (int or tuple): Size of the strides of sliding window
    Returns:
        patches (ndarray): sized[#windows, window_height, window_width]
    '''

    if isinstance(window_size, int):
        h_window, w_window = window_size, window_size
    else:
        h_window, w_window = window_size[0], window_size[1]
    
    if isinstance(strd_size, int):
        h_strd, w_strd = strd_size, strd_size
    else:
        h_strd, w_strd = strd_size[0], strd_size[1]

    h, w = img.shape

    new_shape = (get_size(h, h_window, h_strd), get_size(w, w_window, w_strd), h_window, w_window)
    patches = as_strided(img, shape=new_shape, strides=(w*h_strd, 1*w_strd, w, 1))  # sized[#,#,2,3]
    patches = patches.reshape(-1, h_window, w_window) # sized[#,2,3]
    
    return patches


if __name__ == '__main__':
    img = np.arange(1, 17, dtype=np.uint8).reshape(4, 4)
    # [[ 1  2  3  4]
    #  [ 5  6  7  8]
    #  [ 9 10 11 12]
    #  [13 14 15 16]]

    # img = np.array([img] * 3)
    print(img)
    window = (2, 3)
    strd = (2, 1)
    # patches = sliding_window_3D(img, window, strd)
    patches = sliding_window_2D(img, window, strd)
    print(patches)
    print(patches.shape)