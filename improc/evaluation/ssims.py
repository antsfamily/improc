#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-03-18 21:31:56
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$
from __future__ import division, print_function, absolute_import
import numpy as np
from math import exp
from scipy import ndimage
from improc.common.typevalue import peakvalue, get_drange
from improc.dsp.kernels import convolve
from improc.dsp.filters import sobelfilter


# Kernel that is used in the SSIM implementation presented by the authors in
# "Image Quality Assessment: From Error Visibility to Structural Similarity"
# by Wang et al.
_SSIM_GAUSSIAN_KERNEL_11X11 = np.array([
    [1.0576e-06, 7.8144e-06, 3.7022e-05, 0.00011246, 0.00021905, 0.00027356,
     0.00021905, 0.00011246, 3.7022e-05, 7.8144e-06, 1.0576e-06],
    [7.8144e-06, 5.7741e-05, 0.00027356, 0.00083101, 0.0016186, 0.0020214,
     0.0016186, 0.00083101, 0.00027356, 5.7741e-05, 7.8144e-06],
    [3.7022e-05, 0.00027356, 0.0012961, 0.0039371, 0.0076684, 0.0095766,
     0.0076684, 0.0039371, 0.0012961, 0.00027356, 3.7022e-05],
    [0.00011246, 0.00083101, 0.0039371, 0.01196, 0.023294, 0.029091, 0.023294,
     0.01196, 0.0039371, 0.00083101, 0.00011246],
    [0.00021905, 0.0016186, 0.0076684, 0.023294, 0.045371, 0.056662, 0.045371,
     0.023294, 0.0076684, 0.0016186, 0.00021905],
    [0.00027356, 0.0020214, 0.0095766, 0.029091, 0.056662, 0.070762, 0.056662,
     0.029091, 0.0095766, 0.0020214, 0.00027356],
    [0.00021905, 0.0016186, 0.0076684, 0.023294, 0.045371, 0.056662, 0.045371,
     0.023294, 0.0076684, 0.0016186, 0.00021905],
    [0.00011246, 0.00083101, 0.0039371, 0.01196, 0.023294, 0.029091, 0.023294,
     0.01196, 0.0039371, 0.00083101, 0.00011246],
    [3.7022e-05, 0.00027356, 0.0012961, 0.0039371, 0.0076684, 0.0095766,
     0.0076684, 0.0039371, 0.0012961, 0.00027356, 3.7022e-05],
    [7.8144e-06, 5.7741e-05, 0.00027356, 0.00083101, 0.0016186, 0.0020214,
     0.0016186, 0.00083101, 0.00027356, 5.7741e-05, 7.8144e-06],
    [1.0576e-06, 7.8144e-06, 3.7022e-05, 0.00011246, 0.00021905, 0.00027356,
     0.00021905, 0.00011246, 3.7022e-05, 7.8144e-06, 1.0576e-06]])


def gaussian(winsize, sigma):
    gauss = np.array([exp(-(x - winsize // 2)**2 / float(2 * sigma**2)) for x in range(winsize)])
    return gauss / gauss.sum()


def create_window(winsize, channel):
    _1D_window = gaussian(winsize, 1.5)
    _1D_window = np.reshape(_1D_window, (winsize, 1))
    _2D_window = np.matmul(_1D_window, _1D_window.transpose())

    return _2D_window


def ssim(X, Y, win=None, winsize=11, L=None, k1=0.01, k2=0.03, alpha=1, beta=1, gamma=1, isavg=True, full=False):
    r"""Structural similarity index

    .. math::
           \begin{aligned} l(x, y) &=\frac{2 \mu_{x} \mu_{y}+c_{1}}{\mu_{x}^{2}+\mu_{y}^{2}+c_{1}} \\
           c(x, y) &=\frac{2 \sigma_{x} \sigma_{y}+c_{2}}{\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2}} \\
           s(x, y) &=\frac{\sigma_{x y}+c_{3}}{\sigma_{x} \sigma_{y}+c_{3}} \end{aligned}

    where, :math:`c_1 = (k_1 L)^2, c_2 = (k_2 L)^2, c_3 = c_2 / 2`,
    :math:`L` is the dynamic range of the pixel-values (typically this is :math:`2 ^{\# \text { bits per pixel }}-1`.
    The structure similarity index is expressed as

    .. math::
       \operatorname{SSIM}(x, y)=\left[l(x, y)^{\alpha} \cdot c(x, y)^{\beta} \cdot s(x, y)^{\gamma}\right].

    When :math:`\alpha=\beta=\gamma=1`, SSIM is equal to

    .. math::
       \operatorname{SSIM}(x, y)=\frac{\left(2 \mu_{x} \mu_{y}+c_{1}\right)\left(2 \sigma_{x y}+
       c_{2}\right)}{\left(\mu_{x}^{2}+\mu_{y}^{2}+c_{1}\right)\left(\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2}\right)}

    See "Image Quality Assessment: From Error Visibility to Structural Similarity"
    by Wang et al.

    Parameters
    ----------
    X : {ndarray}
        reconstructed
    Y : {ndarray}
        referenced
    win : {[type]}, optional
        [description] (the default is None, which [default_description])
    winsize : {number}, optional
        [description] (the default is 11, which [default_description])
    L : {integer}, optional
        the dynamic range of the pixel-values (typically this is :math:`2 ^{\# \text { bits per pixel }}-1`. (the default is 255)
    k1 : {number}, optional
        [description] (the default is 0.01, which [default_description])
    k2 : {number}, optional
        [description] (the default is 0.03, which [default_description])
        sizeavg : {bool}, optional
        whether to average (the default is True, which average the result)
    alpha : {number}, optional
        luminance weight (the default is 1)
    beta : {number}, optional
        contrast weight (the default is 1)
    gamma : {number}, optional
        structure weight (the default is 1)
    isavg : {bool}, optional
        IF True, return the average SSIM index of the whole iamge,
    full : {bool}, optional
        IF True, return SSIM, luminance, contrast and structure index (the default is False, which only return SSIM)
    """

    if L is None:
        _, L = get_drange(Y.dtype)

    C1 = (k1 * L)**2
    C2 = (k2 * L)**2
    C3 = C2 / 2.

    if win is None and type(winsize) is not int:
        winsize = 11
        win = _SSIM_GAUSSIAN_KERNEL_11X11
    if win is None and type(winsize) is int:
        win = create_window(winsize, 1)

    muX = convolve(X, win)
    muY = convolve(X, win)
    muXsq = muX * muX
    muYsq = muY * muY

    sigmaXsq = np.abs(convolve(X * X, win) - muXsq)
    sigmaYsq = np.abs(convolve(Y * Y, win) - muYsq)
    sigmaXY = convolve(X * Y, win) - muX * muY

    sigmaX = np.sqrt(sigmaXsq)
    sigmaY = np.sqrt(sigmaYsq)

    luminance = (2. * muX * muY + C1) / (muX * muX + muY * muY + C1)
    contrast = (2 * sigmaX * sigmaY + C2) / (sigmaXsq + sigmaYsq + C2)
    structure = (sigmaXY + C3) / (sigmaX * sigmaY + C3)

    ssim_map = (luminance**alpha) * (contrast**beta) * (structure**gamma)

    if isavg:
        ssim_map = np.mean(ssim_map)
        luminance = np.mean(luminance)
        contrast = np.mean(contrast)
        structure = np.mean(structure)

    if full:
        return ssim_map, luminance, contrast, structure
    else:
        return ssim_map


def gssim(X, Y, win=None, winsize=11, L=None, k1=0.01, k2=0.03, alpha=1, beta=1, gamma=1, isavg=True, full=False):

    if L is None:
        _, L = get_drange(Y.dtype)

    C1 = (k1 * L)**2
    C2 = (k2 * L)**2
    C3 = C2 / 2.

    if win is None and type(winsize) is not int:
        winsize = 11
        win = _SSIM_GAUSSIAN_KERNEL_11X11
    if win is None and type(winsize) is int:
        win = create_window(winsize, 1)

    muX = convolve(X, win)
    muY = convolve(X, win)

    sigmaXsq = np.abs(convolve(X * X, win) - muX * muX)
    sigmaYsq = np.abs(convolve(Y * Y, win) - muY * muY)
    sigmaXY = convolve(X * Y, win) - muX * muY

    sigmaX = np.sqrt(sigmaXsq)
    sigmaY = np.sqrt(sigmaYsq)

    GX = sobelfilter(X)
    GY = sobelfilter(Y)

    muGX = convolve(GX, win)
    muGY = convolve(GY, win)

    sigmaGXsq = np.abs(convolve(GX * GX, win) - muGX * muGX)
    sigmaGYsq = np.abs(convolve(GY * GY, win) - muGY * muGY)
    sigmaGXY = convolve(GX * GY, win) - muGX * muGY


    sigmaGX = np.sqrt(sigmaGXsq)
    sigmaGY = np.sqrt(sigmaGYsq)

    luminance = (2. * muX * muY + C1) / (muX * muX + muY * muY + C1)
    contrast = (2 * sigmaGX * sigmaGY + C2) / (sigmaGXsq + sigmaGYsq + C2)
    structure = (sigmaGXY + C3) / (sigmaGX * sigmaGY + C3)

    gssim_map = (luminance**alpha) * (contrast**beta) * (structure**gamma)

    if isavg:
        gssim_map = np.mean(gssim_map)
        luminance = np.mean(luminance)
        contrast = np.mean(contrast)
        structure = np.mean(structure)

    if full:
        return gssim_map, luminance, contrast, structure
    else:
        return gssim_map


if __name__ == '__main__':

    X = np.random.rand(128, 128)
    Y = np.random.rand(128, 128)

    X = np.zeros((128, 128))
    # Y = np.zeros((128, 128))
    # Y[10:100, 10:100] = 1

    print("-----------------SSIM")

    ssimv = ssim(X, Y, L=1, full=False)
    print(ssimv)

    # default
    ssimv, luminancev, contrastv, structurev = ssim(X, Y, L=1, full=True)
    print(ssimv, luminancev, contrastv, structurev)

    # auto create
    ssimv, luminancev, contrastv, structurev = ssim(X, Y, win=None, winsize=11, L=1, isavg=True, full=True)
    print(ssimv, luminancev, contrastv, structurev)

    # auto create
    ssimv, luminancev, contrastv, structurev = ssim(X, Y, win=None, winsize=3, L=1, isavg=True, full=True)
    print(ssimv, luminancev, contrastv, structurev)

    # not average
    ssimv, luminancev, contrastv, structurev = ssim(X, Y, win=None, winsize=11, L=1, isavg=False, full=True)
    print(ssimv.shape, luminancev.shape, contrastv.shape, structurev.shape)

    print("-----------------GSSIM")


    gssimv = gssim(X, Y, L=1, full=False)
    print(gssimv)

    # default
    gssimv, luminancev, contrastv, structurev = gssim(X, Y, L=1, full=True)
    print(gssimv, luminancev, contrastv, structurev)

    # auto create
    gssimv, luminancev, contrastv, structurev = gssim(X, Y, win=None, winsize=11, L=1, isavg=True, full=True)
    print(gssimv, luminancev, contrastv, structurev)

    # auto create
    gssimv, luminancev, contrastv, structurev = gssim(X, Y, win=None, winsize=3, L=1, isavg=True, full=True)
    print(gssimv, luminancev, contrastv, structurev)

    # not average
    gssimv, luminancev, contrastv, structurev = gssim(X, Y, win=None, winsize=11, L=1, isavg=False, full=True)
    print(gssimv.shape, luminancev.shape, contrastv.shape, structurev.shape)
