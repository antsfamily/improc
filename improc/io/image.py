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
from ..utils.log import *


def imreadadv(filepath, verbose=False):
    """read image data from a file

    read image data from a file

    Parameters
    ----------
    filepath : {string}
        image file path
    verbose : {bool}, optional
        show more information (the default is False.)

    Returns
    -------
    numpy array
        image data array.
    """

    if verbose:
        logging.info("---In imreadadv...")

    ext = os.path.splitext(filepath)[-1]

    if ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        A = tifread(filepath)
    else:
        A = scipyimread(filepath)

    if verbose:
        print(A.shape, filepath)
        logging.info("---Out imreadadv.")

    return A


def imwriteadv(filepath, A):
    """write image to a file

    write image to a file

    Parameters
    ----------
    filepath : {string}
        filepath to be writen
    A : {numpy array}
        Image data to be writen

    Returns
    -------
    number
        0: success
    """

    # logging.info("---In imwriteadv...")

    ext = os.path.splitext(filepath)[-1]

    if ext in ['.tif', '.tiff', '.TIF', '.TIFF']:
        tifsave(filepath, A)
    else:
        A = scipyimsave(filepath)
    # logging.info("---Out imwriteadv.")

    return 0


if __name__ == '__main__':

    imreadadv('/mnt/d/aaa.tif')
