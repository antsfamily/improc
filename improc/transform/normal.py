#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 18:29:48
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import numpy as np
from improc.utils.const import EPS


def normalize(X, mean=None, std=None, axis=None, ver=False):
    r"""normalization

    .. math::
        \bar{X} = \frac{X-\mu}{\sigma}


    Parameters
    ----------
    X : {numpy ndarray}
        data to be normalized,
    mean : {list or None}, optional
        mean value (the default is None, which auto computed)
    std : {list or None}, optional
        standard deviation (the default is None, which auto computed)
    axis : {list or int}, optional
        specify the axis for computing mean value (the default is None, which all elements)
    axis : {bool}, optional
        if True, also return the mean and std (the default is False, which all elements)
    """

    if mean is None:
        if axis is None:
            mean = np.mean(X)
        else:
            mean = np.mean(X, axis, keepdims=True)
    if std is None:
        if axis is None:
            std = np.std(X)
        else:
            std = np.std(X, axis, keepdims=True)

    if ver is True:
        return (X - mean) / std, mean, std
    else:
        return (X - mean) / std


if __name__ == '__main__':

    X = np.random.randn(4, 3, 5, 6)
    # X = th.randn(3, 4)
    XX = normalize(X, axis=(0, 2, 3))
    XX, meanv, stdv = normalize(X, axis=(0, 2, 3), ver=True)
    print(XX.shape)
    print(meanv)
    print(stdv)
