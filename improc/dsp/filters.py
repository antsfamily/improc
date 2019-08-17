#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
from improc.dsp.kernels import convolve, VERTICAL_SOBEL_3x3, HORIZONTAL_SOBEL_3x3


def filtering(X, K):
    pass


def sobelfilter(A, full=False):
    r"""sobel filtering

    filter A with sobel operator

    .. math::
       G_v=\left[\begin{array}{ccc}{-1} & {-2} & {-1} \\ {0} & {0} & {0} \\ {+1} & {+2} & {+1}\end{array}\right] * A

    .. math::
       G_h=\left[\begin{array}{ccc}{-1} & {0} & {+1} \\ {-2} & {0} & {+2} \\ {-1} & {0} & {+1}\end{array}\right] * A

    Parameters
    ----------
    A : {2d or 3d array}
        image to be filtered :math:`H×W×C`.

    full: {bool}
        If True, then return Gh+Gv, Gh, Gv else return Gh+Gv

    Returns
    -------
    G : {2d or 3d array}
        2d-gradient

    Gh : {2d or 3d array}
        gradient in hrizontal

    Gv : {2d or 3d array}
        gradient in vertical

    """

    if np.ndim(A) == 2:
        A = np.pad(A, ((1, 1), (1, 1)), mode="edge")
    if np.ndim(A) == 3:
        A = np.pad(A, ((1, 1), (1, 1), (0, 0)), mode="edge")

    Gh = convolve(A, HORIZONTAL_SOBEL_3x3)
    Gv = convolve(A, VERTICAL_SOBEL_3x3)

    if full:
        return Gh + Gv, Gh, Gv
    else:
        return Gh + Gv