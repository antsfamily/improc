#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
from improc.dsp.filters import imfilter2d
from improc.common.typevalue import get_drange
from improc.transform.preprocessing import scalearr


def imenhance(A, ftype='mean'):

    Af = imfilter2d(A, K=None, ftype=ftype)

    L = get_drange(A.dtype)

    if ftype.find('sharpen') != -1:
        Af = scalearr(A + Af, scaleto=L, scalefrom=L)

    return Af.astype(A.dtype)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.image as img
    import improc as imp

    A = img.imread('../../data/imgs/LenaRGB.tif')
    print(A.shape)

    Ab = imenhance(A, ftype='blur3x3_mean1').astype('uint8')
    As = imenhance(A, ftype='sharpen3x3_type1')
    AbAs = imenhance(Ab, ftype='sharpen3x3_type1')

    print(Ab.min(), Ab.max())
    print(As.min(), As.max())
    print(AbAs.min(), AbAs.max())

    plt.figure()
    plt.subplot(221)
    plt.imshow(A)
    plt.subplot(222)
    plt.imshow(Ab)
    plt.subplot(223)
    plt.imshow(As)
    plt.subplot(224)
    plt.imshow(AbAs)
    plt.tight_layout()
    plt.show()
