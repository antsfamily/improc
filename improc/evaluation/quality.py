#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import os
import math
import numpy as np
import matplotlib.pyplot as plt

"""
Functions to split some images into blocks.

Functions can split different size images randomly or orderly(column-wise or
row-wise):

.. autosummary::
    :nosignatures:

    mse
    psnr
    showorirec
    normalization

"""


def mse(x, y):
    return np.mean((x.astype(float) - y.astype(float)) ** 2)


def psnr(ref, A, Vpeak=None, mode='simple'):
    """
    Peak Signal to Noise Ratio.
        $10 \log10(\frac{V_{peak}^2}{MSE} )$

        For float data, V_{peak} = 1;
        FOr interges, V_{peak} = maxN, e.g. uint8: 255, uint16: 65535 ...


    Parameters
    ----------
    ref : array_like
        Reference data array. For image, it's the original image.
    A : array_like
        The data to be compared. For image, it's the reconstructed image.
    Vpeak : float, int or None, optional
        The peak value. If None, computes automaticly.
    mode : str or None, optional
        'simple' or 'rich'. If Not given or False, just return psnr i.e.
        'simple', else return psnr, mse, Vpeak, imgtype, i.e. 'rich'.

    Returns
    -------
    out : float
        Peak Signal to Noise Ratio value.

    """

    if ref.dtype != A.dtype:
        print("Warning: ref(" + str(ref.dtype) + ")and A(" + str(A.dtype) +
              ")have different type! PSNR may not right!")

    if Vpeak is None:
        if ref.dtype in ('float', 'float16', 'float32', 'float64'):
            Vpeak = 1
        elif ref.dtype in ('uint8', 'uint16', 'uint32', 'uint64'):
            datatype = str(ref.dtype)
            Vpeak = 2 ** int(datatype[4:]) - 1
        elif ref.dtype in ('int64', 'int32', 'int16', 'int8'):
            datatype = str(ref.dtype)
            Vpeak = 2 ** int(datatype[3:]) / 2 - 1
        else:
            raise TypeError('Unrecognized type!')

    MSE = mse(ref, A)
    PSNR = 10 * np.log10((Vpeak ** 2) / MSE)
    if mode is None:
        mode = 'simple'
    if mode == 'rich':
        return PSNR, MSE, Vpeak, ref.dtype
    else:
        return PSNR


# compare
def showorirec(imgs_ori, imgs_rec, mode='rich'):
    for img_ori, img_rec in zip(imgs_ori, imgs_rec):
        cmap = 'Spectral'  # color image map
        if np.ndim(img_ori) == 2:  # Gray
            cmap = 'gray'

        if mode == 'rich':
            print('-----------------------------------------------------------')
            print('Image shape: ', img_ori.shape, img_rec.shape)
            PSNR, MSE, Vpeak, arrdtype = psnr(img_ori, img_rec, mode='rich')

            print('PSNR: ', PSNR, 'dB', 'MSE: ', MSE, 'Vpeak: ',
                  Vpeak, 'Type: ', arrdtype)

        plt.figure()
        plt.subplot(121)
        plt.imshow(img_ori, cmap=cmap)
        plt.title('Original Image')

        plt.subplot(122)
        plt.imshow(img_rec, cmap=cmap)
        plt.title('Reconstructed %.2f' % PSNR + 'dB')
    plt.show()


def normalization(x):
    x = x.astype('float32')
    mu = np.average(x)
    std = np.std(x)
    return (x - mu) / std, mu, std
