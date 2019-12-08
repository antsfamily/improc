#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-31 21:38:26
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$


r"""
Functions for pre-processing images.

Functions used by improc functions:

.. autosummary::
    :nosignatures:

    scalearr
    imgdtype
    normalization
    denormalization

"""


import numpy as np
from improc.utils.const import EPS


def scalearr(X, scaleto=[0, 1], scalefrom=None, istrunc=True, rich=False):
    r"""
    Scale data.

    .. math::
        x \in [a, b] \rightarrow y \in [c, d]

    .. math::
        y = (d-c)*(x-a) / (b-a) + c.

    Parameters
    ----------
    X : array_like
        The data to be scaled.
    scaleto : tuple, list, optional
        Specifies the range of data after beening scaled. Default [0, 1].
    scalefrom : tuple, list, optional
        Specifies the range of data. Default [min(X), max(X)].
    istrunc : bool
        Specifies wether to truncate the data to [a, b], For example,
        If scalefrom == [a, b] and 'istrunc' is true,
        then X[X < a] == a and X[X > b] == b.
    rich : bool
        If you want to see what the data is scaled from and scaled to,
        then you should set it to true
    Returns
    -------
    out : ndarray
        Scaled data array.
    scalefrom, scaleto : list or tuple
        If rich is true, they will also be returned
    """

    if not(isinstance(scaleto, (tuple, list)) and len(scaleto) == 2):
        raise Exception("'scaleto' is a tuple or list, such as (-1,1)")
    if scalefrom is not None:
        if not(isinstance(scalefrom, (tuple, list)) and len(scalefrom) == 2):
            raise Exception("'scalefrom' is a tuple or list, such as (0, 255)")
    else:
        scalefrom = [np.min(X) + 0.0, np.max(X) + 0.0]

    a = scalefrom[0] + 0.0
    b = scalefrom[1] + 0.0
    c = scaleto[0] + 0.0
    d = scaleto[1] + 0.0

    X = X.astype('float')

    if istrunc:
        X[X < a] = a
        X[X > b] = b

    if rich:
        return (X - a) * (d - c) / (b - a + EPS) + c, scalefrom, scaleto
    else:
        return (X - a) * (d - c) / (b - a + EPS) + c


def imgdtype(arr, tdtype=None, isscale=False):
    r"""
    Converts an numpy array to an image data type.

    Parameters
    ----------
    arr : numpy array
        Numpy array that you want to converted.
    dtype : str or None
        The target data type, if not given, does not change. Supports dtype:
        'uint8', 'uint16', 'uint32', 'uint64', 'int64', 'int32', 'int16',
        'int8', 'float', 'float64', 'float32', 'float16'. Note: if float,
        arr will be scaled to [0, 1]
    isscale : bool, or None
        Specifies whether to scale data into the target dtype range, If
        not given, or False, does not scale arr data. This is helpfull
        when some elements in arr are beyond the target dtype range, sets
        isscale to True, will scale data to target dtype range; sets False
        or None, will truncate arr into target dtype range.
        e.g. If target dtype: uint8, and isscale=True,
            then -1 -> 0, 256 -> 255

    Returns
    -------
    arr : numpy array
        Numpy array with target dtype.


    """

    if tdtype is None:
        tdtype = arr.dtype

    if tdtype in ('uint8', 'uint16', 'uint32', 'uint64'):
        datatype = str(tdtype)
        tdtype_range = (0, 2 ** int(datatype[4:]) - 1)
    elif tdtype in ('int64', 'int32', 'int16', 'int8'):
        datatype = str(tdtype)
        tdtype_range = (
            -2 ** int(datatype[3:]) / 2,
            2 ** int(datatype[3:]) / 2 - 1)
    elif tdtype in ('float', 'float64', 'float32', 'float16'):
        tdtype_range = (arr.min(), arr.max())
    else:
        raise TypeError('Unrecognized type!')

    if isscale:
        arr = scalearr(arr, scaleto=tdtype_range)
    else:
        arr[arr < tdtype_range[0]] = tdtype_range[0]
        arr[arr > tdtype_range[1]] = tdtype_range[1]

    return arr.astype(tdtype)


def normalization(X, mod, scalefrom=None, scaleto=None):
    r"""normalization

    data normalization

    Arguments:
        X {data to be normalized} -- [data to be normalized]
        mod {[type]} -- 'zscore', 'minmax'

    Keyword Arguments:
        scalefrom {[type]} -- [description] (default: {None})
        scaleto {[type]} -- [description] (default: {None})
    """

    if mod is 'zscore':
        meanX = np.mean(X)
        stdX = np.std(X)
        X = (X - meanX) / stdX
        return X, meanX, stdX
    elif mod is 'minmax':
        X = scalearr(X, scaleto=[0, 1], scalefrom=None)
        minX = np.min(X)
        maxX = np.max(X)
        # print("=========", minX, maxX)
        return X, minX, maxX
    else:
        X = scalearr(X, scaleto=scaleto, scalefrom=scalefrom)
        return X


def denormalization(X, mod, meanX=None, stdX=None, minX=None, maxX=None):

    if mod is 'zscore':
        meanX = np.mean(X)
        stdX = np.std(X)
        X = (X - meanX) / stdX
        return X, meanX, stdX
    return X


if __name__ == '__main__':

    X = np.random.randn(3, 4)
    print(X)

    print('================zscore=================')
    mod = 'zscore'
    X, meanX, stdX = normalization(X, mod)
    print(X)
    print('==============mean, std================')
    print(meanX, stdX)

    print("=======================================")

    X = np.array([[1, 2, 3], [4, 5, 6]])
    print(X)

    mod = 'zscore'
    # mod = 'minmax'
    X, meanX, stdX = normalization(X, mod)
    print(X)
    print(meanX, stdX)
    print('==============denormalization================')
    X = denormalization(X, mod=mod, meanX=meanX, stdX=stdX)
    print(X)
