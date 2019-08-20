#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
import improc as imp
from skimage import exposure, color
from ..utils.log import *


def convert_colorspace(A, mode=None, verbose=False):
    """convert color sapce

    Valid color spaces are:
    'RGB', 'HSV', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr', 'YDbDr'

    Parameters
    ----------
    A : {ndarray}
        The image (:math:`H×W×C`) to convert, if :math:`C=1`, then do nothing; if :math:`C=3`, do convertion;
        if :math:`C>3`, only the first three channels are processed.
    mode : {str}
        If you want to convert color from space X to Y, just specify
        ``mode`` to ``'X2Y'``, where, ``X, Y`` can be any color spaces mentioned above.
        (default, do nothing)
    verbose : {bool}
        If ``True`` then show log information.

    Returns
    -------
    ndarray
        The converted image.
    """

    if verbose:
        logging.info("---In convert_colorspace...")

    if mode is None:
        if verbose:
            logging.info("---Out convert_colorspace.")
        return A

    p = mode.find('2')
    fromspace = mode[:p]
    tospace = mode[p+1:]

    H = A.copy()  # H-W-C
    if np.ndim(A) == 2:
        if verbose:
            logging.info("~~~Gray image, no need to convert!")

    if np.ndim(A) == 3 and A.shape[2] == 3:
        A = color.convert_colorspace(A, fromspace, tospace)

    if np.ndim(A) == 3 and A.shape[2] > 3:
        A[:, :, 0:3] = color.convert_colorspace(A[:, :, 0:3], fromspace, tospace)

    if verbose:
        logging.info("---Out convert_colorspace.")

    return H
