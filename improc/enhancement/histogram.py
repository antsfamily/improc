#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
import improc as imp
from skimage import exposure


def _histeq_gray(A, nbins=256, mask=None):
    H = exposure.equalize_hist(A, nbins=nbins, mask=mask)
    H = (H * (nbins - 1)).astype(A.dtype)

    return H


def histeq(A, nbins=256, mask=None, mod=None):
    r"""histeq

    Histogram equalization for image with 1, 3, or more channels.

    see https://prateekvjoshi.com/2013/11/22/histogram-equalization-of-rgb-images/

    Parameters
    ----------
    A : {numpy array}
        image data array for processing
    nbins : {number}, optional
        Number of bins for image histogram. Note: this argument is
        ignored for integer images, for which each integer is its own
        bin.
    mask : {ndarray of bools or 0s and 1s}, optional
        [description] (the default is None, which [default_description])
    mod : {str}
        If ``mod`` is ``'eachchannel'``, histogram equalization for each channel of A.
        If ``mod`` is ``None`` and the channel numbers of A is large than 3,
        the first three channel are treated as RGB.

    Returns
    -------
    numpy array
        equalized image array
    """

    H = A.copy()  # H-W-C

    if mod is None:
        if np.ndim(A) == 2:
            # cv2.equalizeHist(A, H)
            H = _histeq_gray(A, nbins=nbins, mask=mask)

        if np.ndim(A) == 3 and A.shape[2] == 3:
            # ycrcb = cv2.cvtColor(H, cv2.COLOR_BGR2YCR_CB)
            # channels = cv2.split(ycrcb)
            # cv2.equalizeHist(channels[0], channels[0])
            # cv2.merge(channels, ycrcb)
            # cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, H)
            H = exposure.equalize_hist(A, nbins=nbins, mask=mask)
        if np.ndim(A) == 3 and A.shape[2] > 3:
            for i in range(A.shape[2]):
                H[:, :, i] = _histeq_gray(A[:, :, i], nbins=nbins, mask=mask)
                # hh = H[:, :, i].copy()
                # cv2.equalizeHist(hh, hh)
                # H[:, :, i] = hh
    if mod is 'eachchannel':
        for i in range(A.shape[2]):
            H[:, :, i] = _histeq_gray(A[:, :, i], nbins=nbins, mask=mask)

    return H


if __name__ == '__main__':

    import cv2

    import matplotlib.pyplot as plt

    im = imp.imreadadv('../../data/imgs/LenaRGB.tif')
    eq = im.copy()

    ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, eq)

    eq1 = exposure.equalize_hist(im)
    eq2 = im.copy()
    eq2[:, :, 0] = imp.histeq(im[:, :, 0])
    eq2[:, :, 1] = imp.histeq(im[:, :, 1])
    eq2[:, :, 2] = imp.histeq(im[:, :, 2])

    plt.figure()
    plt.subplot(221)
    plt.imshow(im)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(eq)
    plt.title('After histeq(OpenCV, RGB)')
    plt.subplot(223)
    plt.imshow(eq1)
    plt.title('After histeq(skimage, RGB)')
    plt.subplot(224)
    plt.imshow(eq2)
    plt.title('After histeq(skimage, each channel)')
    plt.tight_layout()
    plt.show()

    im1o = imp.imreadadv('/mnt/d/DataSets/oi/rsi/RSSRAI2019/our/val/img_2017/image_2017_960_960_1.tif')
    im2o = imp.imreadadv('/mnt/d/DataSets/oi/rsi/RSSRAI2019/our/val/img_2018/image_2018_960_960_1.tif')

    im1 = im1o[:, :, 0:3]
    eq = exposure.equalize_hist(im)
    im1 = imp.scalearr(im1, scaleto=(0, 255), scalefrom=None)
    im1 = im1.astype('uint8')
    eq1 = imp.histeq(im1)

    im2 = im2o[:, :, 0:3]
    im2 = imp.scalearr(im2, scaleto=(0, 255), scalefrom=None)
    im2 = im2.astype('uint8')
    eq2 = imp.histeq(im2)

    plt.figure()
    plt.subplot(221)
    plt.imshow(im1)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(eq1)
    plt.title('After histeq(RGB)')
    plt.subplot(223)
    plt.imshow(im2)
    plt.title('Original')
    plt.subplot(224)
    plt.imshow(eq2)
    plt.title('After histeq(RGB)')
    plt.tight_layout()
    plt.show()

    im1 = im1o[:, :, 3]
    im1 = imp.scalearr(im1, scaleto=(0, 255), scalefrom=None)
    im1 = im1.astype('uint8')
    eq1 = imp.histeq(im1)

    im2 = im2o[:, :, 3]
    im2 = imp.scalearr(im2, scaleto=(0, 255), scalefrom=None)
    im2 = im2.astype('uint8')
    eq2 = imp.histeq(im2)

    plt.figure()
    plt.subplot(221)
    plt.imshow(im1)
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(eq1)
    plt.title('After histeq(the 4-th channel)')
    plt.subplot(223)
    plt.imshow(im2)
    plt.title('Original')
    plt.subplot(224)
    plt.imshow(eq2)
    plt.title('After histeq(the 4-th channel)')
    plt.tight_layout()
    plt.show()

    print("---------")

    im1 = imp.scalearr(im1o, scaleto=(0, 255), scalefrom=None)
    im1 = im1.astype('uint8')
    eq1 = imp.histeq(im1)

    im2 = imp.scalearr(im2o, scaleto=(0, 255), scalefrom=None)
    im2 = im2.astype('uint8')
    eq2 = imp.histeq(im2)

    plt.figure()
    plt.subplot(221)
    plt.imshow(im1[:, :, 0:3])
    plt.title('Original')
    plt.subplot(222)
    plt.imshow(eq1[:, :, 0:3])
    plt.title('After histeq(each channel)')
    plt.subplot(223)
    plt.imshow(im2[:, :, 0:3])
    plt.title('Original')
    plt.subplot(224)
    plt.imshow(eq2[:, :, 0:3])
    plt.title('After histeq(each channel)')
    plt.tight_layout()
    plt.show()
