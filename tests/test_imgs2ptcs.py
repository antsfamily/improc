#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-04 09:43:51
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from improc.blkptcs import imgs2ptcs, showblks
import numpy as np


patchSize = [128, 128, 3]
numPatches = 100

# ----way 1 image pathes

datafolder = '/mnt/d/DataSets/oi/nsi/classical/'
imgspathes = [
    datafolder + 'BaboonRGB.bmp',
    datafolder + 'LenaRGB.bmp',
]

patches = imgs2ptcs(imgspathes, patchSize, numPatches)
print(patches.dtype, patches.shape)
# show
plt.figure(0)
plt.subplot(1, 2, 1)
plt.imshow(patches[:, :, :, 0])
plt.subplot(1, 2, 2)
plt.imshow(patches[:, :, :, numPatches - 2])

imgs = np.uint8(np.random.randint(0, 255, (320, 480, 3, 2)))


patches = imgs2ptcs(imgs, patchSize, numPatches)
print(patches.dtype, patches.shape)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(patches[:, :, 0, 0])
plt.subplot(1, 2, 2)
plt.imshow(patches[:, :, 0, numPatches - 1])


patchSize = [128, 128, 1]
imgspathes = [
    datafolder + 'Baboon.bmp',
    datafolder + 'Lena.bmp',
    datafolder + 'Lena.bmp'
]
patches = imgs2ptcs(imgspathes, patchSize, numPatches)
print(patches.dtype, patches.shape)
# show
plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(patches[:, :, 0, 0], cm.gray)
plt.subplot(1, 2, 2)
plt.imshow(patches[:, :, 0, 0], cm.gray)

plt.show()


showblks(patches, (10, 10), (1, 1), True, 'w', "sample patches")
