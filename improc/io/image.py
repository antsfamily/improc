#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-14 16:14:12
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import os
from scipy.misc import imread as scipyimread
from scipy.misc import imsave as scipyimsave
from .tiff import tifread, tifsave


def imreadadv(filepath, verbose=False):

    ext = os.path.splitext(filepath)[-1]

    if ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        A = tifread(filepath)
    else:
        A = scipyimread(filepath)

    # if verbose:
    print(A.shape, filepath)

    return A


def imswriteadv(filepath, A):
    ext = os.path.splitext(filepath)[-1]

    if ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        tifsave(filepath, A)
    else:
        A = scipyimsave(filepath)


if __name__ == '__main__':

    imreadadv('/mnt/d/aaa.tif')
