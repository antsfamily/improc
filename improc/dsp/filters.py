#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
from improc.dsp.kernels import convolve, VERTICAL_SOBEL_3x3, HORIZONTAL_SOBEL_3x3, KER_DICT


def filtering2d(A, K):
    """filter data

    filter A with kernel K

    Parameters
    ----------
    A : {ndarray}
        data to be filtered
    K : {2d-array}
        kernel for filtering

    Returns
    -------
    ndarray
        filtered data
    """

    if np.ndim(A) == 2:
        A = np.pad(A, ((1, 1), (1, 1)), mode="edge")
    if np.ndim(A) == 3:
        A = np.pad(A, ((1, 1), (1, 1), (0, 0)), mode="edge")

    A = convolve(A, K)

    return A


def imfilter2d(A, K=None, ftype='mean'):

    if K is None:
        K = KER_DICT[ftype]

    A = filtering2d(A, K)

    return A


def sobelfilter(A, gmod='absadd', full=False):
    r"""sobel filtering

    filter A with sobel operator

    .. math::
       G_v=\left[\begin{array}{ccc}{-1} & {-2} & {-1} \\ {0} & {0} & {0} \\ {+1} & {+2} & {+1}\end{array}\right] * A

    .. math::
       G_h=\left[\begin{array}{ccc}{-1} & {0} & {+1} \\ {-2} & {0} & {+2} \\ {-1} & {0} & {+1}\end{array}\right] * A

    .. math::
       G=\sqrt{G_h^2 + G_v^2}

    Parameters
    ----------
    A : {2d or 3d array}
        image to be filtered :math:`H×W×C`.

    gmod: {str}

        - 'add' --> :math:`G=G_h+G_v`,
        - 'absadd' --> :math:`G=|G_h|+|G_v|`,
        - 'squaddsqrt' --> :math:`G=\sqrt{G_h^2 + G_v^2}`,
        - `hvfilt` --> apply sobel filter on hrizontal and vertical
        - `vhfilt` --> apply sobel filter on vertical and hrizontal
        - default `absadd`

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

    if gmod is 'add':
        G = Gh + Gv
    if gmod is 'absadd':
        G = np.abs(Gh) + np.abs(Gv)
    if gmod is 'squaddsqrt':
        G = np.sqrt(Gh * Gh + Gv * Gv)
    if gmod is 'hvfilt':
        G = convolve(Gh, VERTICAL_SOBEL_3x3)
    if gmod is 'vhfilt':
        G = convolve(Gv, HORIZONTAL_SOBEL_3x3)

    if full:
        return G, Gh, Gv
    else:
        return G


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.image as img
    import improc as imp

    A = img.imread('../../data/imgs/LenaRGB.tif')
    print(A.shape)

    Ab = imfilter2d(A, ftype='blur3x3_mean1').astype('uint8')
    As = imfilter2d(A, ftype='sharpen3x3_type1')
    AbAs = imfilter2d(Ab, ftype='sharpen3x3_type1')

    As = A + As
    As = imp.scalearr(As, scaleto=(0, 255), scalefrom=[0, 255]).astype('uint8')

    AbAs = Ab + AbAs
    AbAs = imp.scalearr(AbAs, scaleto=(0, 255), scalefrom=[0, 255]).astype('uint8')

    print(Ab.min(), Ab.max())
    print(As.min(), As.max())
    print(AbAs.min(), AbAs.max())

    plt.figure()
    plt.subplot(221)
    plt.imshow(A)
    plt.subplot(222)
    plt.imshow(Ab)
    plt.subplot(223)
    plt.imshow(As)
    plt.subplot(224)
    plt.imshow(AbAs)
    plt.tight_layout()
    plt.show()

    _, Gh, Gv = sobelfilter(A, full=True)

    G = np.sqrt(Gh * Gh + Gv * Gv)
    Gh = np.abs(Gh)
    Gv = np.abs(Gv)
    G = Gh + Gv
    print(G.shape, G.min(), G.max())
    print(Gh.shape, Gh.min(), Gh.max())
    print(Gv.shape, Gv.min(), Gv.max())

    G_u8 = (255. * G / G.max()).astype('uint8')
    Gh_u8 = (255. * Gh / Gh.max()).astype('uint8')
    Gv_u8 = (255. * Gv / Gv.max()).astype('uint8')

    G_mu_u8 = (255. * (G - G.mean()) / G.max()).astype('uint8')
    Gh_mu_u8 = (255. * (Gh - Gh.mean()) / Gh.max()).astype('uint8')
    Gv_mu_u8 = (255. * (Gv - Gv.mean()) / Gv.max()).astype('uint8')

    plt.figure()
    plt.subplot(221)
    plt.imshow(A)
    plt.title('original')
    plt.subplot(222)
    plt.imshow(Gh)
    plt.title(r'$|G_h|$')
    plt.subplot(223)
    plt.imshow(Gv)
    plt.title(r'$|G_v|$')
    plt.subplot(224)
    plt.imshow(G)
    plt.title(r'$|G_h|+|G_v|$')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(221)
    plt.imshow(A)
    plt.title('original')
    plt.subplot(222)
    plt.imshow(Gh_u8)
    plt.title(r'$|G_h|$, u8')
    plt.subplot(223)
    plt.imshow(Gv_u8)
    plt.title(r'$|G_v|$, u8')
    plt.subplot(224)
    plt.imshow(G_u8)
    plt.title(r'$\sqrt{G_h^2+G_v^2}$, u8')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.subplot(221)
    plt.imshow(A)
    plt.title('original')
    plt.subplot(222)
    plt.imshow(Gh_mu_u8)
    plt.title(r'$|G_h|-\mu_{G_h}$, u8')
    plt.subplot(223)
    plt.imshow(Gv_mu_u8)
    plt.title(r'$|G_v|-\mu_{G_v}$, u8')
    plt.subplot(224)
    plt.imshow(G_mu_u8)
    plt.title(r'$|G_h|+|G_v| - \mu_{|G_h|+|G_v|}$, u8')
    plt.tight_layout()
    plt.show()
