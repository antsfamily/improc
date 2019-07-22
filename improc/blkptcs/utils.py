#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-09-31 21:38:26
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$


from __future__ import absolute_import
import os
import numpy as np
from ..io.image import imreadadv, imwriteadv


def _hw1n2hwn(arr):
    # If arr are H-W-1-N(or H-W-1) ndarray, then convert it to H-W-N.
    if isinstance(arr, np.ndarray):
        arrshape = arr.shape
        if arr.ndim == 4 and arrshape[2] == 1:
            arr = np.reshape(arr, (arrshape[0], arrshape[1], arrshape[3]))
        elif arr.ndim == 3 and arrshape[2] == 1:
            arr = np.reshape(arr, (arrshape[0], arrshape[1]))
    return arr


def _imageinfo(imgpath):
    flag = False
    if os.path.isfile(imgpath):
        img = imreadadv(imgpath)
        # print(img)
        imgdim = np.ndim(img)
        if np.ndim(img) == 2:
            img = img[:, :, np.newaxis]
        flag = True
        return flag, img.dtype, imgdim, img
    else:
        print('Warning: ' + imgpath + ' is not an image file!')
        return flag, None, None, None
