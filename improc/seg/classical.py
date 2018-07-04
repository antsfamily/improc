#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-05-31 08:10:31
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$


import os
import math
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

"""
Image Segmentation functions.

Functions can split different size images randomly or orderly(column-wise or
row-wise):

.. autosummary::
    :nosignatures:

    imgs2bw

"""


def _imageinfo(imgpath):
    flag = False
    print(imgpath)
    if os.path.isfile(imgpath):
        img = imread(imgpath)
        imgdim = np.ndim(img)
        if np.ndim(img) == 2:
            img = img[:, :, np.newaxis]
        flag = True
        return flag, img.dtype, imgdim, img
    else:
        print('Warning: ' + imgpath + ' is not an image file!')
        return flag, None, None, None


def imgs2bw(imgs, Th):
    if Th is None:
        Th = 125

# numpy ndarray H-W-C-N
    if isinstance(imgs, np.ndarray) and np.ndim(imgs) == 4:
        numimgs = np.size(imgs, 3)
        bw = np.zeros(list(imgs.shape), imgs.dtype)
        for i in range(numimgs):
            img = imgs[i, :, :, :]
            img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
            img /= 3
            img[img > Th] = 255
            img[img <= Th] = 0
            bw[i, :, :, 0] = img
            bw[i, :, :, 1] = img
            bw[i, :, :, 2] = img
        return bw
    elif isinstance(imgs, np.ndarray) and np.ndim(imgs) == 3:
        img = imgs
        bw = img
        img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        img /= 3
        img[img > Th] = 255
        img[img <= Th] = 0
        bw[:, :, 0] = img
        bw[:, :, 1] = img
        bw[:, :, 2] = img
        return bw
    # image path list
    elif isinstance(imgs, list):
        numimgs = len(imgs)
        if numimgs == 0:
            return

        # get image data type
        flag, dtype, ndim0, img = _imageinfo(imgs[0])
        if flag:
            bw = np.zeros(list(img.shape) + [numimgs], img.dtype)

        else:
            raise TypeError(
                'bad image pathes list type,'
                'each element of the list should be string')
        i = 0
        for imgpath in imgs:

            flag, _, ndim, img = _imageinfo(imgpath)
            if flag:
                if ndim == ndim0:
                    img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
                    img /= 3
                    img[img > Th] = 255
                    img[img <= Th] = 0
                    bw[:, :, 0] = img
                    bw[:, :, 1] = img
                    bw[:, :, 2] = img
                    i = i + 1
                else:
                    raise TypeError(
                        'I have got images with different' +
                        'ndim:', ndim0, ndim)
            else:
                raise TypeError('Not an image!')
        return bw
        # return patches
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')
