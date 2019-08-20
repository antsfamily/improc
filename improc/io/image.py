#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-14 16:14:12
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import os
# from .tiff import tifread, tifsave
from ..utils.log import *
import skimage.io as io


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

    # ext = os.path.splitext(filepath)[-1]
    # if ext in ['.tif', '.tiff', '.TIF', '.TIFF']:

    A = io.imread(filepath)

    if verbose:
        print(A.shape, filepath)
        logging.info("---Out imreadadv.")

    return A


def imwriteadv(filepath, A, verbose=False):
    """write data to an image file

    write data to an image file

    Parameters
    ----------
    filepath : {string}
        filepath to be writed
    A : {numpy array}
        Image data to be writed

    Returns
    -------
    number
        0: success
    """

    if verbose:
        logging.info("---In imwriteadv...")

    io.imsave(filepath, A)

    if verbose:
        print(A.shape, filepath)
        logging.info("---Out imwriteadv.")


def imsaveadv(filepath, A, verbose=False):
    """save data to an image file

    save data to an image file

    Parameters
    ----------
    filepath : {string}
        filepath to be saved
    A : {numpy array}
        Image data to be saved

    Returns
    -------
    number
        0: success
    """

    if verbose:
        logging.info("---In imsaveadv...")

    io.imsave(filepath, A)

    if verbose:
        print(A.shape, filepath)
        logging.info("---Out imsaveadv.")


if __name__ == '__main__':

    imreadadv('/mnt/d/aaa.tif')
