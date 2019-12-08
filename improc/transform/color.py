#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
import improc as imp
from skimage import exposure, color
from improc.utils.log import *


def convert_colorspace(A, mode=None, drange=None, todrange=False, verbose=False):
    r"""convert color sapce

    Valid color spaces are:
    'RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'

    Parameters
    ----------
    A : {ndarray}
        The image (:math:`H×W×C`) to convert, if :math:`C=1`, then do nothing; if :math:`C=3`, do convertion;
        if :math:`C>3`, only the first three channels are processed.
    mode : {str}
        If you want to convert color from space X to Y, just specify
        ``mode`` to ``'X2Y'``, where, ``X, Y`` can be any color spaces mentioned above.
        (default, do nothing)
    drange : {list or tuple}
        the dynamic range of the pixel-values (typically this is :math:`2 ^{\# \text { bits per pixel }}-1`.)
    todrange : {bool}
        If ``True`` then convert the convert colorspace image to the range of drange.
    verbose : {bool}
        If ``True`` then show log information.

    Returns
    -------
    ndarray
        The converted image.
    """

    if verbose:
        logging.info("---In convert_colorspace...")

    if mode is None:
        if verbose:
            logging.info("---Out convert_colorspace.")
        return A
    dtype = A.dtype
    if drange is None:
        drange = imp.get_drange(dtype)

    p = mode.find('2')
    fromspace = mode[:p]
    tospace = mode[p + 1:]
    # H = A.copy()  # H-W-C
    if np.ndim(A) == 2:
        if verbose:
            logging.info("~~~Gray image, no need to convert!")

    if np.ndim(A) == 3 and A.shape[2] == 3:
        A = color.convert_colorspace(A, fromspace, tospace)
        if todrange:
            A = imp.scalearr(A, scaleto=drange, scalefrom=None, istrunc=False).astype(dtype)

    if np.ndim(A) == 3 and A.shape[2] > 3:
        H = color.convert_colorspace(A[:, :, 0:3], fromspace, tospace)
        if todrange:
            H, sf, st = imp.scalearr(H, scaleto=drange, scalefrom=None, istrunc=False, rich=True)
            A = H[:, :, 0:3].astype(dtype)
    if verbose:
        logging.info("---Out convert_colorspace.")

    return A


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    A = imp.imreadadv('../../data/imgs/LenaRGB.tif')
    A = imp.imreadadv('/mnt/d/DataSets/oi/rsi/RSSRAI2019/our/train/train/img_2017/image_2017_960_960_2.tif')
    print(A.shape)

    A = A * 255. / 1023.
    A = A.astype('uint8')
    # A = A[:, :, 0:3]
    print(A.min(), A.max())
    RGB2HSV = convert_colorspace(A, mode='RGB2HSV', todrange=True, verbose=False)
    RGB2YUV = convert_colorspace(A, mode='RGB2YUV', todrange=True, verbose=False)
    RGB2YCbCr = convert_colorspace(A, mode='RGB2YCbCr', todrange=True, verbose=False)
    RGB2YDbDr = convert_colorspace(A, mode='RGB2YDbDr', todrange=True, verbose=False)
    RGB2YPbPr = convert_colorspace(A, mode='RGB2YPbPr', todrange=True, verbose=False)
    RGB2XYZ = convert_colorspace(A, mode='RGB2XYZ', todrange=True, verbose=False)
    RGB2YIQ = convert_colorspace(A, mode='RGB2YIQ', todrange=True, verbose=False)

    print(RGB2HSV.min(), RGB2HSV.max())
    print(RGB2YUV.min(), RGB2YUV.max())
    print(RGB2YCbCr.min(), RGB2YCbCr.max())
    print(RGB2YDbDr.min(), RGB2YDbDr.max())
    print(RGB2YPbPr.min(), RGB2YPbPr.max())
    print(RGB2XYZ.min(), RGB2XYZ.max())
    print(RGB2YIQ.min(), RGB2YIQ.max())

    plt.figure()
    plt.subplot(331)
    plt.imshow(A)
    plt.subplot(332)
    plt.imshow(RGB2HSV)
    plt.subplot(333)
    plt.imshow(RGB2YUV)
    plt.subplot(334)
    plt.imshow(RGB2YCbCr)
    plt.subplot(335)
    plt.imshow(RGB2YDbDr)
    plt.subplot(336)
    plt.imshow(RGB2YPbPr)
    plt.subplot(337)
    plt.imshow(RGB2XYZ)
    plt.subplot(338)
    plt.imshow(RGB2YIQ)
    plt.tight_layout()
    plt.show()
