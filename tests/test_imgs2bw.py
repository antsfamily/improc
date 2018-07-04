#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-04 09:43:51
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import matplotlib.cm as cm
from matplotlib import pyplot as plt
import improc as imp


datafolder = '/mnt/d/DataSets/oi/nsi/classical/'
imgspathes = [
    datafolder + 'BaboonRGB.bmp',
    datafolder + 'LenaRGB.bmp',
]
print(imgspathes)
bws = imp.imgs2bw(imgspathes, 50)
print(bws.dtype, bws.shape)

print(bws)

plt.figure()

plt.imshow(bws[:, :, :, 0], cm.gray)

plt.show()
