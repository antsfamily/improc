#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-07-31 13:38:45
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from improc.blkptcs import imgs2blks, showblks
import numpy as np


blksize = [64, 64, 3]

# ----way 1 image pathes
imgspathes = ['/mnt/d/DataSets/oi/nsi/classical/BaboonRGB.bmp',
              '/mnt/d/DataSets/oi/nsi/classical/LenaRGB.bmp']
# imgspathes = ['/home/liu/data/goldenfish.jpg']

blks, imgsshape = imgs2blks(imgspathes, blksize, 'symmetric')
print(blks.dtype, blks.shape)
print(imgsshape)
# show
plt.figure(0)
plt.subplot(1, 2, 1)
plt.imshow(blks[:, :, :, 0])
plt.subplot(1, 2, 2)
plt.imshow(blks[:, :, :, -1])

showblks(blks, (10, 10), (1, 1), True, 'w', "Blocks")


imgs = np.uint8(np.random.randint(0, 255, (320, 480, 3, 2)))
imgs[18:blksize[0] - 8, 18:blksize[1] - 8, :, :] = 0


blks, imgsshape = imgs2blks(imgs, blksize, 'symmetric')
print(blks.dtype, blks.shape)
print(imgsshape)
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(blks[:, :, :, 0])
plt.subplot(1, 2, 2)
plt.imshow(blks[:, :, :, -1])


plt.show()
