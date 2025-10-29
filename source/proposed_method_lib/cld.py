import numpy as np
import math
from scipy.fftpack import dct, idct

# Code for CSD feature descriptor, presented in: https://link.springer.com/chapter/10.1007/978-3-662-46742-8_24

def divide_img_blocks(img, n_blocks=(8, 8)):
    horizontal = np.array_split(img, n_blocks[0])
    splitted_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]

    block_size = (math.floor(img.shape[0] / n_blocks[0]), math.floor(img.shape[1] / n_blocks[1]))

    return np.asarray(splitted_img, dtype=np.ndarray).reshape((n_blocks[0], n_blocks[1], block_size[0], block_size[1], img.shape[2]))

def calculate_avg_value_in_block(block, axis=(0,1)):
	return np.mean(block, axis=axis)

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def array2d_to_zigzag(array):
	return np.concatenate([np.diagonal(array[::-1,:], k)[::(2*(k % 2)-1)] for k in range(1-array.shape[0], array.shape[0])])