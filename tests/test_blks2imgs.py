#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-16 17:34:25
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from improc.blkptcs import imgs2blks, blks2imgs

# ----------------------------  RGB  --------------------------------
# -------------------------------------------------------------------
#      imgs     to         blks
blksize = [128, 128, 3]

# ----way 1 image pathes Color RGB
datafolder = '/mnt/d/DataSets/oi/nsi/classical/'
imgspathes = [
    datafolder + 'BaboonRGB.bmp',
    datafolder + 'LenaRGB.bmp',
]

outfolder = '/mnt/d/test/'

# -------------------------------------------------------------------
#       imgs     to       blks
blks, imgsshape = imgs2blks(imgspathes, blksize, 'symmetric')
print(blks.dtype, blks.shape)
print(imgsshape)
# show
plt.figure(0)
plt.subplot(1, 2, 1)
plt.imshow(blks[:, :, :, 0])
plt.subplot(1, 2, 2)
plt.imshow(blks[:, :, :, -1])
# -------------------------------------------------------------------
#      blks     to         imgs

imgs = blks2imgs(blks, imgsshape, None, outfolder)

if type(imgs) == np.ndarray:
    print(imgs.dtype, imgs.shape)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(imgs[:, :, :, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(imgs[:, :, :, -1])
plt.show()

# -----------------------------Gray----------------------------------
# -------------------------------------------------------------------
blksize = [128, 128, 1]
imgspathes = [
    '/mnt/d/DataSets/oi/nsi/classical/Baboon.bmp',
    '/mnt/d/DataSets/oi/nsi/classical/Lena.bmp',
    # '/mnt/d/DataSets/oi//nsi/classical/Lena.bmp'
]
# -------------------------------------------------------------------
#       imgs     to       blks
blks, imgsshape = imgs2blks(imgspathes, blksize, 'symmetric')
print(blks.dtype, blks.shape)
print(imgsshape)
# show
plt.figure(0)
plt.subplot(1, 2, 1)
plt.imshow(blks[:, :, 0, 0], cm.gray)
plt.subplot(1, 2, 2)
plt.imshow(blks[:, :, 0, -1], cm.gray)
plt.show()
# -------------------------------------------------------------------
#      blks     to         imgs
imgs = blks2imgs(blks, imgsshape)

if type(imgs) == np.ndarray:
    print(imgs.dtype, imgs.shape)
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgs[:, :, 0, 0], cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(imgs[:, :, 0, -1], cm.gray)

    plt.show()
