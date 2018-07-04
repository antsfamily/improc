#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-04 09:43:51
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import numpy as np
import improc as imp

arr = np.array([[0, 128], [255, 100]], dtype='uint8')

print('Before scaling:')
print(arr)

# [0, 255] --> [0, 1]
scaleto = [0, 1]
print('scale to ' + str(scaleto) + ' :')
print(imp.scalearr(arr, scaleto=scaleto))

# [0, 255] --> [0.2, 0.8]
scaleto = [0.2, 0.8]
print('scale to ' + str(scaleto) + ' :')
print(imp.scalearr(arr, scaleto=scaleto))

print(arr, arr.dtype)

arr = np.array([[0.1, 0.5], [0.99, 0.3]], dtype='float')

print('------------------------')
print('Before scaling:')
print(arr)


# [min, max] --> [0, 255]
scaleto = [0, 255]
print('scale from [min, max] to ' + str(scaleto) + ' :')
print(imp.scalearr(arr, scaleto=scaleto))

# [0, 1] --> [0, 255]
scaleto = [0, 255]
scalefrom = [0, 1]
print('scale from ' + str(scalefrom) + ' to ' + str(scaleto) + ' :')
print(imp.scalearr(arr, scaleto=scaleto, scalefrom=scalefrom))


# [0.2, 0.8] --> [0, 255]
scaleto = [0, 255]
scalefrom = [0.2, 0.8]
print('scale from ' + str(scalefrom) + ' to ' + str(scaleto) + ' :')
print(imp.scalearr(arr, scaleto=scaleto, scalefrom=scalefrom))

# [0.2, 0.8] --> [0, 255]
scaleto = [0, 255]
scalefrom = [0.2, 0.8]
print('scale from ' + str(scalefrom) + ' to ' + str(scaleto) + '(truncated) :')
print(imp.scalearr(arr, scaleto=scaleto, scalefrom=scalefrom, istrunc=False))


print('----------------------------')
arr = np.array([[-1 / 255.0 + 0.6, 1 / 512.0],
                [0.02, -1 / 1024.0], [-1 / 1026.0, 0.0]])
print(arr)
# [-1, 1] --> [0, 255]
scaleto = [0, 255]
scalefrom = [-1, 1]
print('scale from ' + str(scalefrom) + ' to ' + str(scaleto) + '(truncated) :')
arr = imp.scalearr(arr, scaleto=scaleto, scalefrom=scalefrom, istrunc=False)

print(arr)
arr = np.round(arr)
print(arr)


# [-1, 1] --> [0, 255]
scaleto = [-1, 1]
scalefrom = [0, 255]
print('scale from ' + str(scalefrom) + ' to ' + str(scaleto) + '(truncated) :')
print(imp.scalearr(arr, scaleto=scaleto, scalefrom=scalefrom, istrunc=False))
