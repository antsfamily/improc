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
from ..common.typevalue import peakvalue
from ..utils.log import *


r"""
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


def mse(o, r):
    r"""Mean Squared Error

    The Mean Squared Error (MSE) is expressed as

    .. math::
        {\rm MSE} = \frac{1}{MN}\sum_{i=1}^{M}\sum_{j=0}^{N}[|{\bm I}(i,j)|, |\hat{\bm I}(i, j)|]^2

    Arguments
    ---------------
    o : ndarray
        Orignal signal matrix.

    r : ndarray
        Reconstructed signal matrix

    Returns
    ---------------
    MSE : float
        Mean Squared Error

    """

    return np.mean(np.square((np.abs(o).astype(float) - np.abs(r).astype(float))))


def psnr(ref, A, Vpeak=None, mode='simple'):
    r"""Peak Signal-to-Noise Ratio

    The Peak Signal-to-Noise Ratio (PSNR) is expressed as

    .. math::
        {\rm PSNR} = 10 \log10(\frac{V_{peak}^2}{\rm MSE})

    For float data, :math:`V_{peak} = 1`;

    For interges, :math:`V_{peak} = 2^{nbits}`,
    e.g. uint8: 255, uint16: 65535 ...


    Parameters
    -----------
    o : array_like
        Reference data array. For image, it's the original image.
    r : array_like
        The data to be compared. For image, it's the reconstructed image.
    Vpeak : float, int or None, optional
        The peak value. If None, computes automaticly.
    mode : str or None, optional
         'simple' or 'rich'. 'simple' (default) --> just return psnr i.e.
         'rich' --> return psnr, mse, Vpeak, imgtype.

    Returns
    -------
    PSNR : float
        Peak Signal to Noise Ratio value.

    """

    if ref.dtype != A.dtype:
        print("Warning: ref(" + str(ref.dtype) + ")and A(" + str(A.dtype) +
              ")have different type! PSNR may not right!")

    if Vpeak is None:
        peakvalue(ref, ref.dtype)

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
