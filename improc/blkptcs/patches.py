#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-09-31 21:38:26
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$


from __future__ import absolute_import
import os
import math
import random
import numpy as np
# from scipy.misc import imread,
from ..io.image import imreadadv, imwriteadv
import matplotlib.pyplot as plt
from ..transform.preprocessing import scalearr, imgdtype

from .utils import _hw1n2hwn, _imageinfo
from ..utils.log import *

from ..common.randomfunc import randperm2d


r"""
Functions to sample some images into patches.

Functions can sample different size images randomly:

.. autosummary::
    :nosignatures:

    imgs2ptcs
    imgsAB2ptcs
    selptcs
    geocluptcs

"""


def _cmpNumSamples(numptcs, numImages):
    numSamples = []
    temp = np.int(np.floor(numptcs / numImages))
    rest = np.mod(numptcs, numImages)
    for i in range(0, numImages):
        numSamples.append(temp)
    if rest != 0:
        numSamples[random.randint(0, numImages - 1)] += rest
    return numSamples


def sampleimg(imgA, ptcsize, numSamples, imgB=None, imgC=None, seed=None):
    imgSize = imgA.shape
    np.random.seed(seed)

    P = np.reshape(np.array(range(0, imgSize[0] * imgSize[1])), imgSize[0:2])[0:imgSize[0] - ptcsize[0], 0:imgSize[1] - ptcsize[1]]
    ys, xs = randperm2d(imgSize[0], imgSize[1], numSamples, P)
    if imgSize[2] == ptcsize[2]:
        zs = [0] * numSamples
    else:
        zs = random.randint(0, imgSize[2] - ptcsize[2], numSamples)

    if imgB is None:
        patches1imgA = np.zeros(ptcsize + [numSamples], imgA.dtype)

        for s, y, x, z in zip(range(numSamples), ys, xs, zs):
            patches1imgA[:, :, :, s] = imgA[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
        return patches1imgA
    if imgC is None:
        patches1imgA = np.zeros(ptcsize + [numSamples], imgA.dtype)
        patches1imgB = np.zeros(ptcsize + [numSamples], imgB.dtype)

        for s, y, x, z in zip(range(numSamples), ys, xs, zs):
            patches1imgA[:, :, :, s] = imgA[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
            patches1imgB[:, :, :, s] = imgB[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
        return patches1imgA, patches1imgB
    else:
        patches1imgA = np.zeros(ptcsize + [numSamples], imgA.dtype)
        patches1imgB = np.zeros(ptcsize + [numSamples], imgB.dtype)
        patches1imgC = np.zeros(ptcsize + [numSamples], imgC.dtype)

        for s, y, x, z in zip(range(numSamples), ys, xs, zs):
            patches1imgA[:, :, :, s] = imgA[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
            patches1imgB[:, :, :, s] = imgB[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
            patches1imgC[:, :, :, s] = imgC[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]

        return patches1imgA, patches1imgB, patches1imgC


def imgs2ptcs(imgs, ptcsize, numptcs, seed=None):
    r"""
    Sample diffrent patches from imgs.

    Parameters
    ----------
    imgs : {array_like, or list of image pathes}.
        Images to be sampled, a H-W-C-N numpy ndarray, or a list of image
        pathes with diffrent image shapes.
    ptcsize : {int tuple, list or None, optional}
        Specifies the each patch size (rows, cols, channel) that you want to
        sampled. If not given, ptcsize=[8, 8, 1].
    numptcs : {int or None, optional}
        The number of patches that you want to sample, if None, numptcs=100.
    seed : {int or None, optional}
        Random seed, default None(differ each time.

    Returns
    -------
    ptcs : {ndarray}
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == ptcsize.

    See Also
    --------
    imgs2blks, blks2imgs, showblks.


    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from imgblk import imgs2ptcs, showblks
    >>> ptcsize = [128, 128, 3]
    >>> numptcs = 20
    >>> imgspathes = ['/mnt/d/DataSets/oi/nsi/classical/BaboonRGB.bmp',
                     '/mnt/d/DataSets/oi/nsi/classical/LenaRGB.bmp',]
    >>> ptcs = imgs2ptcs(imgspathes, ptcsize , numptcs)
    >>> print(ptcs.dtype, ptcs.shape)
    >>> # show
    >>> showblks(ptcs, rcsize=(10,10))

    See "test_imgs2ptcs.py", "test_imgs2blks.py", "test_blks2imgs.py",
    "test_showblks.py" for more Examples.
    """
    if ptcsize is None:
        ptcsize = (8, 8, 1)
    if numptcs is None:
        numptcs = 100
    ptcsize = list(ptcsize)

    logging.info("---In imgs2ptcs...")
    # numpy ndarray H-W-C-N
    if isinstance(imgs, np.ndarray) and np.ndim(imgs) == 4:
        numimgs = np.size(imgs, 3)
        numSamples = _cmpNumSamples(numptcs, numimgs)

        ptcs = np.zeros(ptcsize + [numptcs], imgs.dtype)
        cpos = 0
        for i in range(numimgs):
            ptcs[:, :, :, cpos:cpos + numSamples[i]] = sampleimg(
                imgs[:, :, :, i], ptcsize, numSamples[i], seed=seed)
            cpos += numSamples[i]

        return ptcs
    # image path list
    elif isinstance(imgs, list):
        numimgs = len(imgs)
        if numimgs == 0:
            logging.info("---Out imgs2ptcs...")
            return None
        else:
            numSamples = _cmpNumSamples(numptcs, numimgs)
        # get image data type
        flag, dtype, ndimA, _ = _imageinfo(imgs[0])
        if flag:
            ptcs = np.zeros(ptcsize + [numptcs], dtype)

        else:
            raise TypeError(
                'bad image pathes list type,'
                'each element of the list should be string')
        i = 0
        cpos = 0
        for imgpath in imgs:
            flag, _, ndim, img = _imageinfo(imgpath)
            print(img.shape, imgpath)
            if flag:
                if ndim == ndimA:
                    ptcs[:, :, :, cpos:cpos + numSamples[i]] = sampleimg(
                        img, ptcsize, numSamples[i], seed=seed)
                    cpos += numSamples[i]
                    i = i + 1
                else:
                    raise TypeError(
                        'I have got images with different' +
                        'ndim:', ndimA, ndim)
            else:
                raise TypeError('Not an image!')
        logging.info("---Out imgs2ptcs...")

        return ptcs
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')
    logging.info("---Out imgs2ptcs...")


def imgsAB2ptcs(imgsA, imgsB, ptcsize, numptcs, seed=None):
    r"""
    Sampling diffrent patches from imgsA imgsB.

    Parameters
    ----------
    imgsA/B : {array_like, or list of image pathes.}
        Images to be sampled, a H-W-C-N numpy ndarray, or a list of image
        pathes with diffrent image shapes.
    ptcsize : {int tuple, list or None, optional}
        Specifies the each patch size (rows, cols, channel) that you want to
        sampled. If not given, ptcsize=[8, 8, 1].
    numptcs : {int or None, optional}
        The number of patches that you want to sample, if None, numptcs=100.
    seed : {int or None, optional}
        Random seed, default None(differ each time.

    Returns
    -------
    ptcsA : {ndarray}
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == ptcsize.

    ptcsB : {ndarray}
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == ptcsize.
    See Also
    --------
    imgs2blks, blks2imgs, showblks.


    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from imgblk import imgs2ptcs, showblks
    >>> ptcsize = [128, 128, 3]
    >>> numptcs = 20
    >>> imgspathesA = ['/mnt/d/DataSets/oi/nsi/classical/BaboonRGB.bmp']
        imgspathesB = ['/mnt/d/DataSets/oi/nsi/classical/LenaRGB.bmp']
    >>> ptcsA, ptcsB = imgsAB2ptcs(imgspathesA, imgspathesB, ptcsize , numptcs)
    >>> print(ptcsA.dtype, ptcsA.shape)
    >>> # show
    >>> showblks(ptcsA, rcsize=(10,10))

    See "test_imgs2ptcs.py", "test_imgs2blks.py", "test_blks2imgs.py",
    "test_showblks.py" for more Examples.

    """

    logging.info("---In imgsAB2ptcs...")

    if ptcsize is None:
        ptcsize = (8, 8, 1)
    if numptcs is None:
        numptcs = 100

    # numpy ndarray H-W-C-N
    if isinstance(imgsA, np.ndarray) and np.ndim(imgsA) == 4:
        numimgs = np.size(imgsA, 3)
        numSamples = _cmpNumSamples(numptcs, numimgs)

        ptcsA = np.zeros(ptcsize + [numptcs], imgsA.dtype)
        ptcsB = np.zeros(ptcsize + [numptcs], imgsB.dtype)
        cpos = 0
        for i in range(numimgs):
            ptcA, ptcB = sampleimg(imgsA[:, :, :, i], ptcsize, numSamples[
                i], imgsB[:, :, :, i], seed=seed)
            ptcsA[:, :, :, cpos:cpos + numSamples[i]] = ptcA
            ptcsB[:, :, :, cpos:cpos + numSamples[i]] = ptcB
            cpos += numSamples[i]
        logging.info("---Out imgsAB2ptcs...")
        return ptcsA, ptcsB

    # image path list
    elif isinstance(imgsA, list):
        numimgs = len(imgsA)
        if numimgs == 0:
            logging.info("~~~No Images!")
            logging.info("---Out imgsAB2ptcs...")
            return None
        else:
            numSamples = _cmpNumSamples(numptcs, numimgs)
        # get image data type
        flag, dtype, ndimA, _ = _imageinfo(imgsA[0])
        if flag:
            ptcsA = np.zeros(ptcsize + [numptcs], dtype)
            ptcsB = np.zeros(ptcsize + [numptcs], dtype)

        else:
            raise TypeError(
                'bad image pathes list type,'
                'each element of the list should be string')

        cpos = 0
        for n in range(numimgs):
            imgpathA = imgsA[n]
            imgpathB = imgsB[n]
 #           print(imgpathA, imgpathB)
            flag, _, ndim, imgB = _imageinfo(imgpathB)
            flag, _, ndim, imgA = _imageinfo(imgpathA)
            if flag:
                if ndim == ndimA:

                    ptcA, ptcB = sampleimg(imgA, ptcsize, numSamples[
                        n], imgB, seed=seed)
                    ptcsA[:, :, :, cpos:cpos + numSamples[n]] = ptcA
                    ptcsB[:, :, :, cpos:cpos + numSamples[n]] = ptcB
                    cpos += numSamples[n]
                else:
                    raise TypeError(
                        'I have got images with different' +
                        'ndim:', ndimA, ndim)
            else:
                raise TypeError('Not an image!')
        logging.info("---Out imgsAB2ptcs...")

        return ptcsA, ptcsB
        # return patches
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')


def imgsABC2ptcs(imgsA, imgsB, imgsC, ptcsize, numptcs, seed=None):
    r"""
    Sampling diffrent patches from imgsA imgsB imgsC.

    Parameters
    ----------
    imgsA/B/C : {array_like, or list of image pathes}.
        Images to be sampled, a H-W-C-N numpy ndarray, or a list of image
        pathes with diffrent image shapes.
    ptcsize : {int tuple, list or None, optional}
        Specifies the each patch size (rows, cols, channel) that you want to
        sampled. If not given, ptcsize=[8, 8, 1].
    numptcs : {int or None, optional}
        The number of patches that you want to sample, if None, numptcs=100.
    seed : {int or None, optional}
        Random seed, default None(differ each time.

    Returns
    -------
    ptcsA : {ndarray}
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == ptcsize.

    ptcsB : {ndarray}
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == ptcsize.
    ptcsC : {ndarray}
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == ptcsize.
    See Also
    --------
    imgs2blks, blks2imgs, showblks.


    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from imgblk import imgs2ptcs, showblks
    >>> ptcsize = [128, 128, 3]
    >>> numptcs = 20
    >>> imgspathesA = ['/mnt/d/DataSets/oi/nsi/classical/BaboonRGB.bmp']
        imgspathesB = ['/mnt/d/DataSets/oi/nsi/classical/LenaRGB.bmp']
    >>> ptcsA, ptcsB = imgsAB2ptcs(imgspathesA, imgspathesB, ptcsize , numptcs)
    >>> print(ptcsA.dtype, ptcsA.shape)
    >>> # show
    >>> showblks(ptcsA, rcsize=(10,10))

    See "test_imgs2ptcs.py", "test_imgs2blks.py", "test_blks2imgs.py",
    "test_showblks.py" for more Examples.

    """

    logging.info("---IN imgsABC2ptcs...")

    if ptcsize is None:
        ptcsize = (8, 8, 1)
    if numptcs is None:
        numptcs = 100
    # numpy ndarray H-W-C-N
    if isinstance(imgsA, np.ndarray) and np.ndim(imgsA) == 4:
        numimgs = np.size(imgsA, 3)
        numSamples = _cmpNumSamples(numptcs, numimgs)

        ptcsA = np.zeros(ptcsize + [numptcs], imgsA.dtype)
        ptcsB = np.zeros(ptcsize + [numptcs], imgsB.dtype)
        ptcsC = np.zeros(ptcsize + [numptcs], imgsC.dtype)

        cpos = 0
        for i in range(numimgs):
            # print(i, numimgs)
            # print(imgsA.shape, imgsB.shape, imgsC.shape)
            ptcA, ptcB, ptcC = sampleimg(imgsA[:, :, :, i],
                                         ptcsize, numSamples[i],
                                         imgsB[:, :, :, i],
                                         imgsC[:, :, :, i], seed=seed)
            # print(ptcA.shape, ptcB.shape, ptcC.shape)
            # print(i, numSamples[i], ptcA.shape, ptcB.shape, ptcC.shape)
            ptcsA[:, :, :, cpos:cpos + numSamples[i]] = ptcA
            ptcsB[:, :, :, cpos:cpos + numSamples[i]] = ptcB
            ptcsC[:, :, :, cpos:cpos + numSamples[i]] = ptcC
            cpos += numSamples[i]

        logging.info("---Out imgsABC2ptcs...")
        return ptcsA, ptcsB, ptcsC
    # image path list
    elif isinstance(imgsA, list):
        numimgs = len(imgsA)
        if numimgs == 0:
            logging.info("~~~No images!")
            logging.info("---Out imgsABC2ptcs...")

            return None
        else:
            numSamples = _cmpNumSamples(numptcs, numimgs)
        # get image data type
        flag, dtype, ndimA, _ = _imageinfo(imgsA[0])
        if flag:
            ptcsA = np.zeros(ptcsize + [numptcs], dtype)
            ptcsB = np.zeros(ptcsize + [numptcs], dtype)

        else:
            raise TypeError(
                'bad image pathes list type,'
                'each element of the list should be string')

        cpos = 0
        for n in range(numimgs):
            imgpathA = imgsA[n]
            imgpathB = imgsB[n]
            imgpathC = imgsC[n]
            flag, _, ndimA, imgA = _imageinfo(imgpathA)
            flag, _, ndimB, imgB = _imageinfo(imgpathB)
            flag, _, ndimC, imgC = _imageinfo(imgpathC)
            if flag:
                if (ndimC == ndimA) and (ndimB == ndimA) and (ndimB == ndimC):

                    ptcA, ptcB, ptcC = sampleimg(
                        imgA, ptcsize, numSamples[n], imgB, imgC, seed=seed)
                    ptcsA[:, :, :, cpos:cpos + numSamples[n]] = ptcA
                    ptcsB[:, :, :, cpos:cpos + numSamples[n]] = ptcB
                    ptcsC[:, :, :, cpos:cpos + numSamples[n]] = ptcC
                    cpos += numSamples[n]
                else:
                    raise TypeError(
                        'I have got images with different dim %f, %f, %f '
                        % (ndimA, ndimB, ndimC))
            else:
                raise TypeError('Not an image!')

        logging.info("---Out imgsAB2ptcs...")

        return ptcsA, ptcsB, ptcsC
        # return patches
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')


def selptcs(patches, numsel=None, method=None, thresh=None, sort=None):
    r"""
    Selects some patches based on std, var...

    Parameters
    ----------
    patches : {array_like}
        Image patches, a pH-pW-pC-pN numpy ndarray.
    numsel : {int, float or None, optional}
        How many patches you want to select. If integer, then returns numsel
        patches; If float in [0,1], then numsel*100 percent of the patches
        will be returned. If not given, does not affect.
    method : {str or None, optional}
        Specifies which method to used for evaluating each patch. Option:
        'std'(standard deviation), 'var'(variance),
        the first numsel patch scores will be returned.
        If not given, does nothing.
    thresh : {float, optional}
        When method are specified, using thresh as the threshold, and those
        patches whose scores are larger(>=) then thresh will be returned.
    sort : {str or None, optional}
        Sorts the patches according to pathes scores in sort way('descent',
        'ascent'). If not given, does not affect, i.e. origional order.

    Returns
    -------
    selpat : {ndarray}
        A pH-pW-pC-pNx numpy ndarray.
    idxsel : {ndarray}
        A pNx numpy array, indicates the position of selpat in patches.
    selptcs_scores: {ndarray}
        A pNx numpy array, indicates the scores of selpat in patches, with
        measurements specfied by method.

    See Also
    --------
    imgs2ptcs, imgs2blks, blks2imgs, showblks.

    Examples
    --------
    >>> patches = np.uint8(np.random.randint(0, 255, (8, 8, 3, 101)))
    >>> # just compute scores:
    >>> selpat, _, scores = selptcs(patches, method='std')
    >>> # compute scores and sort by scores:
    >>> selpat, _, scores = selptcs(patches, method='std', sort='descent')
    >>> # compute scores and sort by scores, and just select scores > thresh
    >>> selpat, idxsel, _ = selptcs(patches, method='std', thresh=15)
    >>> # compute scores and sort by scores, and select scores > thresh, and
    >>> # sort in descending order.
    >>> selpat, idxsel, _ = selptcs(patches, method='std', thresh=15,
        sort='descent')
    >>>


    """

    logging.info("---In selptcs...")

    if isinstance(patches, np.ndarray):
        pH, pW, pC, pN = patches.shape
    else:
        raise TypeError('"patches" should be a pH-pW-pC-pN numpy ndarray.')

    if numsel is None:
        numsel = pN
    if isinstance(numsel, float) and (numsel >= 0 and numsel <= 1):
        numsel = int(numsel * pN)

    idxsel = np.array(range(0, pN))
    # check method
    if method is None:
        logging.info("---Out selptcs.")
        return patches, idxsel[0:min(numsel, pN)], None
    elif method == 'var':
        selptcs_scores = np.reshape(patches, (pH * pW * pC, pN))  # pH*pW*pC-pN
        selptcs_scores = np.var(selptcs_scores, 0)             # (pN,)
    elif method == 'std':
        selptcs_scores = np.reshape(patches, (pH * pW * pC, pN))  # pH*pW*pC-pN
        selptcs_scores = np.std(selptcs_scores, 0)             # (pN,)
    else:
        raise ValueError('method' + ' does not support now!')

    if thresh is not None:
        idxsel = idxsel[selptcs_scores >= thresh]
        selptcs_scores = selptcs_scores[idxsel]

    if sort == 'ascent':
        idxsel = np.argsort(selptcs_scores, 0)
        selptcs_scores = selptcs_scores[idxsel]
    elif sort == 'descent':
        idxsel = np.argsort(selptcs_scores, 0)
        idxsel = idxsel[::-1]
        selptcs_scores = selptcs_scores[idxsel]
    elif sort is None:
        idxsel = np.array(range(0, min(numsel, pN)))
        selptcs_scores = selptcs_scores[idxsel]
    else:
        raise ValueError('"sort" should be ascent, descent or None!')
    idxsel = idxsel[0:min(numsel, pN)]

    logging.info("---Out selptcs.")

    return patches[:, :, :, idxsel], idxsel, selptcs_scores


def geocluptcs(patches, TH=None, Rstar=None):
    r"""

    Classifies patches into smooth blocks and rough blocks

    Parameters
    ----------
    patches : {array_like}
        Image patches, a pH-pW-pC-pN numpy ndarray.
    TH : {int, float or None, optional}
        Threshold for .
    Rstar : {int, float or None, optional}
        Specifies which method to used for evaluating each patch. Option:
        'std'(standard deviation), 'var'(variance),

    Returns
    -------
    selpat : {ndarray}
        A pH-pW-pC-pNx numpy ndarray.
    idxsel : {ndarray}
        A pNx numpy array, indicates the position of selpat in patches.
    selptcs_scores: {ndarray}
        A pNx numpy array, indicates the scores of selpat in patches, with
        measurements specfied by method.

    See Also
    --------
    imgs2ptcs, imgs2blks, blks2imgs, showblks.

    Examples
    --------
    >>> patches = np.uint8(np.random.randint(0, 255, (8, 8, 3, 101)))
    >>> # just compute scores:
    >>> selpat, _, scores = geocluptcs(patches, method='std')
    >>> # compute scores and sort by scores:
    >>> selpat, _, scores = selptcs(patches, method='std', sort='descent')
    >>> # compute scores and sort by scores, and just select scores > thresh
    >>> selpat, idxsel, _ = selptcs(patches, method='std', thresh=15)
    >>> # compute scores and sort by scores, and select scores > thresh, and
    >>> # sort in descending order.
    >>> selpat, idxsel, _ = selptcs(patches, method='std', thresh=15,
        sort='descent')
    >>>


    """

    logging.info("---In geocluptcs...")

    if not (isinstance(patches, np.ndarray) and patches.ndim == 4):
        raise TypeError('"patches" should be a pH-pW-pC-pN numpy array!')
    pH, pW, pC, pN = patches.shape
    sortptcs, sortptcs_idx, sortptcs_scores = selptcs(
        patches,
        method='std', sort='descent')
    if TH is None:
        TH = np.mean(sortptcs_scores) / 2

    # split smooth block and rough block
    smoothptcs_idx = sortptcs_idx[sortptcs_scores < TH]
    smoothptcs = patches[:, :, :, smoothptcs_idx]
    roughptcs_idx = sortptcs_idx[sortptcs_scores >= TH]
    roughptcs = patches[:, :, :, roughptcs_idx]
    rpN = np.size(roughptcs, 3)
    # compute the gradient of rough blocks
    if pC == 1:  # gray patches
        # dh: pH-pW-rpN, given arr with type uint8, If g<0, then np.gradient
        #   return wrong (e.g.,-255 -> 1, -127.5 -> 0.5), so, convert to float
        #   , or scale to [0,1]
        dh, dw, _ = np.gradient(_hw1n2hwn(roughptcs).astype(np.float32))
        G = np.concatenate((
            dh.reshape(pH * pW * pC, rpN),
            dw.reshape(pH * pW * pC, rpN)), axis=1)
        G = G.reshape(pH * pW * pC, 2, rpN)
    elif pC > 1:
        dh, dw, dc, _ = np.gradient(roughptcs.astype(np.float32))
        G = np.concatenate((
            dh.reshape(pH * pW * pC, rpN),
            dw.reshape(pH * pW * pC, rpN)), axis=1)
        G = np.concatenate((
            G, dc.reshape(pH * pW * pC, rpN)), axis=1)
        G = G.reshape(pH * pW * pC, 3, rpN)
    # compute Singular vector and value
    R = []
    Angle = []
    eps = 1e-10
    for k in range(0, rpN):
        u, s, v = np.linalg.svd(G[:, :, k])  # 64*2 or 192*3
        R.append((s[0] - s[1]) / (s[0] + s[1] + eps))
        Angle.append(np.arctan(v[0, 1] / v[0, 0]))
    R = np.array(R)
    Angle = np.array(Angle)
    # classify rough patches
    dominantptcs = []
    dominantptcs_idx = []
    if Rstar is None:
        Rstar = 0.2
    stochasticptcs_idx = roughptcs_idx[R < Rstar]
    stochasticptcs = patches[:, :, :, stochasticptcs_idx]
    dom_idx = roughptcs_idx[R >= Rstar]
    dom_angle = Angle[R >= Rstar]  # (-pi/2, pi/2)

    # divide into 0, pi/6, pi/3 pi/2(-pi/2), 2pi/3(-pi/3), 5pi/6(-pi/6),
    # so delta = pi/12
    delta = np.pi / 12
    # 0 rad patches
    dominantptcs_idx.append(dom_idx[abs(dom_angle) < delta])
    dominantptcs_idx.append(dom_idx[abs(dom_angle - np.pi / 6) <= delta])
    dominantptcs_idx.append(dom_idx[abs(dom_angle - np.pi / 3) < delta])
    dominantptcs_idx.append(
        dom_idx[abs((abs(dom_angle) - np.pi / 2)) <= delta])
    dominantptcs_idx.append(dom_idx[abs(dom_angle + np.pi / 3) < delta])
    dominantptcs_idx.append(dom_idx[abs(dom_angle + np.pi / 6) <= delta])

    for k in range(0, 6, 1):

        if len(dominantptcs_idx[k]) != 0:
            dominantptcs.append(patches[:, :, :, dominantptcs_idx[k]])
        else:
            dominantptcs.append(np.array([]))
    if len(smoothptcs_idx) == 0:
        smoothptcs = np.array([])
    if len(stochasticptcs_idx) == 0:
        stochasticptcs = np.array([])

    logging.info("---Out geocluptcs.")

    return smoothptcs, dominantptcs, stochasticptcs,\
        smoothptcs_idx, dominantptcs_idx, stochasticptcs_idx
