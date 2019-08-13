#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-09-31 21:38:26
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$


from __future__ import absolute_import
import math
import numpy as np
# from scipy.misc import imread,
from ..io.image import imreadadv, imwriteadv
import matplotlib.pyplot as plt
from ..transform.preprocessing import scalearr, imgdtype
from ..utils.log import *


r"""
Functions to visualize blocks or patches.

Functions can visualize 4-D numpy array:

.. autosummary::
    :nosignatures:

    showblks
    showfilters

"""


def showblks(blks, rcsize=None, stride=None, plot=True, bgcolor='w', cmap=None, title=None, xlabel=None, ylabel=None):
    r"""
    Trys to show image blocks in one image.

    Parameters
    ----------
    blks : {array_like}
        Blocks to be shown, a bH-bW-bC-bN numpy ndarray.
    rcsize : {int tuple or None, optional}
        Specifies how many rows and cols of blocks that you want to show,
        e.g. (rows, cols). If not given, rcsize=(rows, clos) will be computed
        automaticly.
    stride : {int tuple or None, optional}
        The step size (blank pixels nums) in row and col between two blocks.
        If not given, stride=(1,1).
    plot : {bool, optional}
        True for ploting, False for silent and returns a H-W-C numpy ndarray
        for showing.
    bgcolor : {char or None, optional}
        The background color, 'w' for white, 'k' for black. Default, 'w'.

    Returns
    -------
    out : {ndarray or bool}
        A H-W-C numpy ndarray for showing.

    See Also
    --------
    imgs2ptcs, imgs2blks, blks2imgs.

    Examples
    --------
    >>> blks = np.uint8(np.random.randint(0, 255, (8, 8, 3, 101)))
    >>> showblks(blks, bgcolor='k')

    """

    logging.info("---In showblks...")

    if plot is None:
        plot = True
    # print(blks.min(), blks.max(), "'''", blks.shape, np.mean(blks))
    if blks.size == 0:
        return blks
    if not (isinstance(blks, np.ndarray) and blks.ndim == 4):
        raise TypeError('"blks" should be a pH-pW-pC-pN numpy array!')

    _, _, _, bN = blks.shape

    if rcsize is None:
        rows = int(math.sqrt(bN))
        cols = int(bN / rows)
        if bN % cols > 0:
            rows = rows + 1
    else:
        rows = rcsize[0]
        cols = rcsize[1]
    # step size
    if stride is None:
        stride = (1, 1)
    # background color
    if bgcolor == 'w':
        bgcolor_value = 255
    elif bgcolor == 'k':
        bgcolor_value = 0
    if bN < rows * cols:
        A = np.pad(blks,
                   ((0, stride[0]), (0, stride[1]),
                       (0, 0), (0, rows * cols - bN)),
                   'constant', constant_values=bgcolor_value)
    else:
        A = np.pad(blks,
                   ((0, stride[0]), (0, stride[1]), (0, 0), (0, 0)),
                   'constant', constant_values=bgcolor_value)
        A = A[:, :, :, 0:rows * cols]

    aH, aW, aC, aN = A.shape

    A = np.transpose(A, (3, 0, 1, 2)).reshape(
        rows, cols, aH, aW, aC).swapaxes(
        1, 2).reshape(rows * aH, cols * aW, aC)

    aH, aW, aC = A.shape
    A = A[0:aH - stride[0], 0:aW - stride[1], :]
    if aC == 1:
        A = A[:, :, 0]

    if title is None:
        title = 'Show ' + str(bN) + ' blocks in ' + str(rows) + \
            ' rows ' + str(cols) + ' cols, with stride' + str(stride)
    if plot:
        plt.figure()
        plt.imshow(A, cmap=cmap)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
    logging.info("---Out showblks.")

    return A    # H-W-C


def showfilters(filters, fsize=None, rcsize=None, stride=None, plot=None,
                bgcolor='w', title=None, isscale=True):
    r"""
    Trys to show weight filters in one image.

    Parameters
    ----------
    filters : {array_like}
        Weights to be shown, a fdim-by-numf numpy ndarray.
    fsize : {int tuple list or None, optional}
        Specifies the height and width of one filter, e.g. (hf, wf, cf). If not
        given, fsize=(hf, wf, cf) will be computed automaticly, but may wrong.
    rcsize : {int tuple list or None, optional}
        Specifies how many rows and cols of filters that you want to show,
        e.g. (rows, cols). If not given, rcsize=(rows, clos) will be computed
        automaticly.
    stride : {int tuple or None, optional}
        The step size (blank pixels nums) in row and col between two filters.
        If not given, stride=(1,1).
    plot : {bool or None, optional}
        True for ploting, False for silent and returns a H-W-C numpy ndarray
        for showing. If not given, plot = True.
    bgcolor : {char or None, optional}
        The background color, 'w' for white, 'k' for black. Default, 'w'.
    title : {str or None}
        If plot, title is used to specify the title.
    isscale : {bool or None, optional}
        If True, scale to [0, 255], else does nothing.

    Returns
    -------
    out : {ndarray or bool}
        A H-W-C numpy ndarray for showing.

    See Also
    --------
    imgs2ptcs, imgs2blks, blks2imgs.

    Examples
    --------
    >>> filters = np.uint8(np.random.randint(0, 255, (192, 101)))
    >>> showfilters(filters, bgcolor='k')

    """

    logging.info("---In showfilters...")

    if filters is None:
        raise ValueError('You should give me filters')

    if not (isinstance(filters, np.ndarray) and np.ndim(filters) == 2):
        raise TypeError('filters should be a fdim-by-numfs numpy array!')

    fdim, numfs = filters.shape
    if fsize is None:
        cf = 1
        if math.sqrt(fdim) != int(math.sqrt(fdim)):  # not square
            if fdim % 3 == 0:   # 3 channels
                cf = 3
        hwf = fdim / cf
        hf = int(math.sqrt(hwf))
        wf = int(hwf / hf)
        if hwf % wf > 0:
            hf = hf + 1
        fsize = (hf, wf, cf)
        if np.prod(fsize) != fdim:
            raise ValueError("I can't inference the size of each filter!")

    elif not(isinstance(fsize, (tuple, list)) and len(fsize) == 3):
        raise ValueError('fsize should be a tuple list, like (8, 10, 3)')

    if isscale:
        filters = scalearr(filters, (0, 255))
    filters = filters.reshape(
        (fsize[2], fsize[0], fsize[1], numfs)).transpose(1, 2, 0, 3)

    logging.info("---Out showfilters.")

    return showblks(filters, rcsize=rcsize, stride=stride,
                    plot=plot, bgcolor=bgcolor, title=title)
