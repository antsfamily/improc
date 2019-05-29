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
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
from ..utils.prep import scalearr, imgdtype

"""
Functions to split some images into blocks.

Functions can split different size images randomly or orderly(column-wise or
row-wise):

.. autosummary::
    :nosignatures:

    imgs2ptcs
    imgsAB2ptcs
    selptcs
    geocluptcs
    imgs2blks
    blks2imgs
    showblks
    showfilters

"""

# np.random.seed(2016)


def _hw1n2hwn(arr):
    # If arr are H-W-1-N(or H-W-1) ndarray, then convert it to H-W-N.
    if isinstance(arr, np.ndarray):
        arrshape = arr.shape
        if arr.ndim == 4 and arrshape[2] == 1:
            arr = np.reshape(arr, (arrshape[0], arrshape[1], arrshape[3]))
        elif arr.ndim == 3 and arrshape[2] == 1:
            arr = np.reshape(arr, (arrshape[0], arrshape[1]))
    return arr


def _imageinfo(imgpath):
    flag = False
    if os.path.isfile(imgpath):
        img = imread(imgpath)
        # print(img)
        imgdim = np.ndim(img)
        if np.ndim(img) == 2:
            img = img[:, :, np.newaxis]
        flag = True
        return flag, img.dtype, imgdim, img
    else:
        print('Warning: ' + imgpath + ' is not an image file!')
        return flag, None, None, None


def _cmpNumSamples(numptcs, numImages):
    numSamples = []
    temp = np.int(np.floor(numptcs / numImages))
    rest = np.mod(numptcs, numImages)
    for i in range(0, numImages):
        numSamples.append(temp)
    if rest != 0:
        numSamples[random.randint(0, numImages - 1)] += rest
    return numSamples


def sampleimg(img, ptcsize, numSamples, imgB=None):
    imgSize = img.shape
#    print("====:", img.shape, imgB.shape)
    if imgB is None:
        patches1img = np.zeros(ptcsize + [numSamples], img.dtype)

        for s in range(numSamples):
            y = random.randint(0, imgSize[0] - ptcsize[0])
            x = random.randint(0, imgSize[1] - ptcsize[1])
            z = random.randint(0, imgSize[2] - ptcsize[2])
            patches1img[:, :, :, s] = img[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
        return patches1img
    else:
        patches1img = np.zeros(ptcsize + [numSamples], img.dtype)
        patches1imgB = np.zeros(ptcsize + [numSamples], img.dtype)
#	print("====:", patches1img.shape, patches1imgB.shape)
#       print(img.shape, imgB.shape)
        for s in range(numSamples):
            y = random.randint(0, imgSize[0] - ptcsize[0])
            x = random.randint(0, imgSize[1] - ptcsize[1])
            z = random.randint(0, imgSize[2] - ptcsize[2])
#            print("+++",x, y, z)
#            print("-----",y, y + ptcsize[0], x, x + ptcsize[1], z, z + ptcsize[2])
            patches1img[:, :, :, s] = img[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
            patches1imgB[:, :, :, s] = imgB[
                y:y + ptcsize[0], x:x + ptcsize[1], z:z + ptcsize[2]]
        return patches1img, patches1imgB


def imgs2ptcs(imgs, ptcsize, numptcs):
    """
    Sample patches from imgs.

    Parameters
    ----------
    imgs : array_like, or list of image pathes.
        Images to be sampled, a H-W-C-N numpy ndarray, or a list of image
        pathes with diffrent image shapes.
    ptcsize : int tuple, list or None, optional
        Specifies the each patch size (rows, cols, channel) that you want to
        sampled. If not given, ptcsize=[8, 8, 1].
    numptcs : int or None, optional
        The number of patches that you want to sample, if None, numptcs=100.

    Returns
    -------
    ptcs : ndarray
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

# numpy ndarray H-W-C-N
    if isinstance(imgs, np.ndarray) and np.ndim(imgs) == 4:
        numimgs = np.size(imgs, 3)
        numSamples = _cmpNumSamples(numptcs, numimgs)

        ptcs = np.zeros(ptcsize + [numptcs], imgs.dtype)
        for i in range(numimgs):

            ptcs[:, :, :, i * numSamples[i]:
                 (i + 1) * numSamples[i]] = sampleimg(
                imgs[:, :, :, i], ptcsize, numSamples[i])
        return ptcs
    # image path list
    elif isinstance(imgs, list):
        numimgs = len(imgs)
        if numimgs == 0:
            return
        else:
            numSamples = _cmpNumSamples(numptcs, numimgs)
        # get image data type
        flag, dtype, ndim0, _ = _imageinfo(imgs[0])
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
                if ndim == ndim0:
                    ptcs[:, :, :, cpos:cpos + numSamples[i]] = sampleimg(
                        img, ptcsize, numSamples[i])
                    cpos += numSamples[i]
                    i = i + 1
                else:
                    raise TypeError(
                        'I have got images with different' +
                        'ndim:', ndim0, ndim)
            else:
                raise TypeError('Not an image!')
        return ptcs
        # return patches
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')


def imgsAB2ptcs(imgsA, imgsB, ptcsize, numptcs):
    """
    Sampling patches from imgsA imgsB.

    Parameters
    ----------
    imgsA/B : array_like, or list of image pathes.
        Images to be sampled, a H-W-C-N numpy ndarray, or a list of image
        pathes with diffrent image shapes.
    ptcsize : int tuple, list or None, optional
        Specifies the each patch size (rows, cols, channel) that you want to
        sampled. If not given, ptcsize=[8, 8, 1].
    numptcs : int or None, optional
        The number of patches that you want to sample, if None, numptcs=100.

    Returns
    -------
    ptcsA : ndarray
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == ptcsize.

    ptcsB : ndarray
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
        for i in range(numimgs):
            ptcA, ptcB = sampleimg(imgsA[:, :, :, i], ptcsize, numSamples[
                                   i], imgsB[:, :, :, i])
            ptcsA[:, :, :, i * numSamples[i]:(i + 1) * numSamples[i]] = ptcA
            ptcsB[:, :, :, i * numSamples[i]:(i + 1) * numSamples[i]] = ptcB

        return ptcsA, ptcsB
    # image path list
    elif isinstance(imgsA, list):
        numimgs = len(imgsA)
        if numimgs == 0:
            return
        else:
            numSamples = _cmpNumSamples(numptcs, numimgs)
        # get image data type
        flag, dtype, ndim0, _ = _imageinfo(imgsA[0])
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
                if ndim == ndim0:

                    ptcA, ptcB = sampleimg(imgA, ptcsize, numSamples[n], imgB)
                    ptcsA[:, :, :, cpos:cpos + numSamples[n]] = ptcA
                    ptcsB[:, :, :, cpos:cpos + numSamples[n]] = ptcB
                    cpos += numSamples[n]
                else:
                    raise TypeError(
                        'I have got images with different' +
                        'ndim:', ndim0, ndim)
            else:
                raise TypeError('Not an image!')
        return ptcsA, ptcsB
        # return patches
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')


def selptcs(patches, numsel=None, method=None, thresh=None, sort=None):
    """
    Selects some patches based on std, var...

    Parameters
    ----------
    patches : array_like
        Image patches, a pH-pW-pC-pN numpy ndarray.
    numsel : int, float or None, optional
        How many patches you want to select. If integer, then returns numsel
        patches; If float in [0,1], then numsel*100 percent of the patches
        will be returned. If not given, does not affect.
    method : str or None, optional
        Specifies which method to used for evaluating each patch. Option:
        'std'(standard deviation), 'var'(variance),
        the first numsel patch scores will be returned.
        If not given, does nothing.
    thresh : float, optional
        When method are specified, using thresh as the threshold, and those
        patches whose scores are larger(>=) then thresh will be returned.
    sort : str or None, optional
        Sorts the patches according to pathes scores in sort way('descent',
        'ascent'). If not given, does not affect, i.e. origional order.

    Returns
    -------
    selpat : ndarray
        A pH-pW-pC-pNx numpy ndarray.
    idxsel : ndarray
        A pNx numpy array, indicates the position of selpat in patches.
    selptcs_scores: ndarray
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

    return patches[:, :, :, idxsel], idxsel, selptcs_scores


def geocluptcs(patches, TH=None, Rstar=None):
    r"""

    Classifies patches into smooth blocks and rough blocks

    Parameters
    ----------
    patches : array_like
        Image patches, a pH-pW-pC-pN numpy ndarray.
    TH : int, float or None, optional
        Threshold for .
    Rstar : int, float or None, optional
        Specifies which method to used for evaluating each patch. Option:
        'std'(standard deviation), 'var'(variance),

    Returns
    -------
    selpat : ndarray
        A pH-pW-pC-pNx numpy ndarray.
    idxsel : ndarray
        A pNx numpy array, indicates the position of selpat in patches.
    selptcs_scores: ndarray
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

    return smoothptcs, dominantptcs, stochasticptcs,\
        smoothptcs_idx, dominantptcs_idx, stochasticptcs_idx


def _padhwc(imgsize, blksize):
    # computes pad h w c
    hpad = np.mod(imgsize[0], blksize[0])
    wpad = np.mod(imgsize[1], blksize[1])
    cpad = np.mod(imgsize[2], blksize[2])
    if hpad > 0:
        hpad = blksize[0] - hpad
    if wpad > 0:
        wpad = blksize[1] - wpad
    if cpad > 0:
        cpad = blksize[2] - cpad
    return hpad, wpad, cpad


def padimgs(imgs, pad_width, mode):
    pimgsorig = np.pad(imgs, pad_width, mode)
    pimgsize = pimgsorig.shape
    if np.ndim(imgs) == 3:
        pimgs = np.zeros(
            (pimgsize[0], pimgsize[1], pimgsize[2], 1), imgs.dtype)
        pimgs[:, :, :, 0] = pimgsorig
        return pimgs
    else:
        return pimgsorig
# splits padded images into blocks


def pimgs2blks(pimgs, blksize):
    """
        blks: bH-bW-bC-bN, img1_blk1_bc1, img1_blk1_bc2, img1_blk1_bc3,
        img1_blk2_bc1, img1_blk2_bc2, img1_blk2_bc3, ...

    """
    pH, pW, pC, pN = pimgs.shape  # (H+0)-by-(W+0)-by-Channels-by-numImages
    mnBlock = np.array([pH, pW, pC]) / np.array(blksize)
    numBlks1 = np.int(np.prod(mnBlock))
    blks = np.zeros(
        [blksize[0], blksize[1], blksize[2], pN * numBlks1], pimgs.dtype)
    cnt = 0
    for n in range(0, pN):
        for i in range(0, pH, blksize[0]):
            for j in range(0, pW, blksize[1]):
                for k in range(0, pC, blksize[2]):
                    blks[:, :, :, cnt] = pimgs[
                        i:blksize[0] + i,
                        j:blksize[1] + j,
                        k:blksize[2] + k, n]
                    # if cnt < 6:
                    # 	plt.figure(cnt)
                    # 	plt.imshow(blks[:,:,:,cnt])
                    cnt = cnt + 1
    return blks


def imgs2blks(imgs, blksize=[8, 8, 3], padmode='symmetric'):
    """
    blks, imgsShape = imgs2blks(imgs, blksize) trys to split the imgs(an H-W-C
    -N numpy ndarray, or list of images filepathes) into image blocks(an bH-bW-
    bC-bN ndarray) orderly; imgsShape contains each image's size in imgs for
    fighting back to images.

    Trys to split some images into blocks orderly (1.channel-wisely, 2.colum-
    wisely, 3.row-wisely).

    Parameters
    ----------
    imgs : array_like, or list of image pathes.
        Images to be splited, a H-W-C-N numpy ndarray, or a list of image
        pathes with diffrent image shapes.
    blksize : int tuple, list or None, optional
        Specifies the each block size (rows, cols, channel) that you want to
        split. If not given, blksize=[8, 8, 3].
    padmode : str or None, optional
        Padding mode when an image can't be split fully. 'symmetric' or 'edge'
        can be choosen, see np.pad for more option. The Channel will also be
        pad if it is not enough, e.g., for an H-W-1(gray) image, if you want
        to get blksize=[8,8,3].

    Returns
    -------
    blks : ndarray
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == blksize.
    imgsshape : tuple
        A tuple contains the shape of each image in imgs. Even if all images
        have the same size, imgsshape will still contain each image size like
        ((H, W, C), (H, W, C)).

    See Also
    --------
    imgs2ptcs, imgs2blks, blks2imgs.

    Examples
    --------
    >>> blks, imgsInfo = imgs2blks(imgspathes, [8, 8, 1], 'symmetric')

"""
    # numpy ndarray H-W-C-N
    if isinstance(imgs, np.ndarray) and np.ndim(imgs) == 4:
        imgsShape = imgs.shape
        # compute padding params
        hpad, wpad, cpad = _padhwc(imgsShape, blksize)
        # pads images
        pimgs = padimgs(imgs, ((0, hpad), (0, wpad), (0, cpad), (0, 0)),
                        padmode)
        # splits padded images into blocks bH-bW-bC-bN,
        #   where bN = bN1+bN2+...+bNn
        # bN1: blks of image_1, bNn blks of image_n
        return pimgs2blks(pimgs, blksize), imgsShape
    # image path list
    elif isinstance(imgs, list):
        if len(imgs) == 0:
            return
        # process the first image
        numimgs = 0
        flag, datatype, _, img = _imageinfo(imgs[0])
        if not flag:
            raise TypeError('No an avaliable image!')
            return
        imgsShape = []
        # blks[:,:,:,0], for concatenate, will be removed later
        blks = np.zeros((blksize[0], blksize[1], blksize[2], 1), datatype)

        # split the rest images
        for imgpath in imgs:
            flag, _, _, img = _imageinfo(imgpath)
            if flag:  # yes, it's an image
                imgsShape.append(img.shape)
                # compute padding params
                hpad, wpad, cpad = _padhwc(imgsShape[numimgs], blksize)
                # pads images
                pimgs = padimgs(
                    img, ((0, hpad), (0, wpad), (0, cpad)), padmode)
                # splits padded images into blocks bH-bW-bC-bN,
                #   where bN = bN1+bN2+...+bNn
                # bN1: blks of image_1, bNn blks of image_n
                blks = np.concatenate(
                    (blks, pimgs2blks(pimgs, blksize)), axis=3)
                numimgs = numimgs + 1
        if numimgs != 0:
            return blks[:, :, :, 1:], imgsShape  # blks[:,:,:,0] all zeros
        else:
            raise TypeError('No avaliable image!')
        # return patches
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')


def _get_pimgshape(imgShape, blkSize):
    hpad, wpad, cpad = _padhwc(imgShape, blkSize)
    pH = imgShape[0] + hpad
    pW = imgShape[1] + wpad
    pC = imgShape[2] + cpad
    return (pH, pW, pC)


def _rmsingledim(arr):
    if arr.shape[2] == 1:  # Gray
        arr = np.reshape(arr, arr.shape[0:2])
    return arr


def blks2imgs(blks, imgsShape, index=None, tofolder=None):
    """
    Fight image blocks back to images.

    Parameters
    ----------
    blks : array_like
        A bH-bW-bC-bN numpy ndarray, where, bH, bW, bC, bN specify height,
        width, channels, number of images respectivelly. If blks Contains
        Gray Images and RGB Images, then bC = 3, and Gray Images is copied.
    imgsShape : tuple or list
        Specify each image's size.
    index : int, optional
        The image that you want to fight back.
    tofolder : str (path), optional
        Save specified images into files: image_0.png, image_i.png, ... ,
        image_N.png

    Returns
    -------
    out : ndarray or bool
        A list of image numpy ndarray(H-W-C).

    See Also
    --------
    imgs2ptcs, imgs2blks.

    """
    if np.ndim(blks) != 4:
        raise TypeError(
            "blks must be a bH-bW-bC-bN 4-dimentional numpy ndarray!")

    if len(imgsShape) == 0:
        raise ValueError("You must tell me images's shape in blks!")

    if type(imgsShape) in (list, tuple):
        numimgs = len(imgsShape)
        if index is None:  # all image
            index = range(0, numimgs)

    if tofolder is not None and (not os.path.isdir(tofolder)):
        raise TypeError('"tofolder" should be an available path!')

    imgvaluetype = blks.dtype
    imgsPos = [0]
    blksSize = blks.shape
    bH, bW, bC, bN = blksSize
    blkSize = [bH, bW, bC]

    for imgShape in imgsShape:
        pH, pW, pC = _get_pimgshape(imgShape, blkSize)
        mnBlock = np.array([pH, pW, pC]) / np.array([bH, bW, bC])
        numblks1 = np.prod(mnBlock)
        imgsPos.append(imgsPos[-1] + numblks1)

    numFightBack = len(index)

    imgsShapeFightBack = []
    for idx in index:
        imgsShapeFightBack.append(imgsShape[idx])

    imgs = []
    for n in range(0, numFightBack):
        imgShape = imgsShapeFightBack[n]
        pH, pW, pC = _get_pimgshape(imgShape, blkSize)
        arr = np.zeros([pH, pW, pC], imgvaluetype)
        cnt = int(imgsPos[index[n]])
        for i in range(0, pH, bH):
            for j in range(0, pW, bW):
                for k in range(0, pC, bC):
                    arr[i:bH + i, j:bW + j, k:bC + k] = blks[:, :, :, cnt]
                    cnt = cnt + 1

        img = _rmsingledim(arr[0:imgShape[0], 0:imgShape[1], 0:imgShape[2]])
        imgs.append(img)

        if tofolder is not None:
            imsave(tofolder + '/image_' + str(n) + '.png', img)

    return imgs  # list contains N images numpy ndarray


def showblks(blks,
             rcsize=None, stride=None, plot=True, bgcolor='w', cmap=None, title=None, xlabel=None, ylabel=None):
    """
    Trys to show image blocks in one image.

    Parameters
    ----------
    blks : array_like
        Blocks to be shown, a bH-bW-bC-bN numpy ndarray.
    rcsize : int tuple or None, optional
        Specifies how many rows and cols of blocks that you want to show,
        e.g. (rows, cols). If not given, rcsize=(rows, clos) will be computed
        automaticly.
    stride : int tuple or None, optional
        The step size (blank pixels nums) in row and col between two blocks.
        If not given, stride=(1,1).
    plot : bool, optional
        True for ploting, False for silent and returns a H-W-C numpy ndarray
        for showing.
    bgcolor : char or None, optional
        The background color, 'w' for white, 'k' for black. Default, 'w'.

    Returns
    -------
    out : ndarray or bool
        A H-W-C numpy ndarray for showing.

    See Also
    --------
    imgs2ptcs, imgs2blks, blks2imgs.

    Examples
    --------
    >>> blks = np.uint8(np.random.randint(0, 255, (8, 8, 3, 101)))
    >>> showblks(blks, bgcolor='k')

    """

    if plot is None:
        plot = True

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

    return A    # H-W-C


def showfilters(filters, fsize=None, rcsize=None, stride=None, plot=None,
                bgcolor='w', title=None, isscale=True):
    """
    Trys to show weight filters in one image.

    Parameters
    ----------
    filters : array_like
        Weights to be shown, a fdim-by-numf numpy ndarray.
    fsize : int tuple list or None, optional
        Specifies the height and width of one filter, e.g. (hf, wf, cf). If not
        given, fsize=(hf, wf, cf) will be computed automaticly, but may wrong.
    rcsize : int tuple list or None, optional
        Specifies how many rows and cols of filters that you want to show,
        e.g. (rows, cols). If not given, rcsize=(rows, clos) will be computed
        automaticly.
    stride : int tuple or None, optional
        The step size (blank pixels nums) in row and col between two filters.
        If not given, stride=(1,1).
    plot : bool or None, optional
        True for ploting, False for silent and returns a H-W-C numpy ndarray
        for showing. If not given, plot = True.
    bgcolor : char or None, optional
        The background color, 'w' for white, 'k' for black. Default, 'w'.
    title : str or None
        If plot, title is used to specify the title.
    isscale : bool or None, optional
        If True, scale to [0, 255], else does nothing.

    Returns
    -------
    out : ndarray or bool
        A H-W-C numpy ndarray for showing.

    See Also
    --------
    imgs2ptcs, imgs2blks, blks2imgs.

    Examples
    --------
    >>> filters = np.uint8(np.random.randint(0, 255, (192, 101)))
    >>> showfilters(filters, bgcolor='k')

    """

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
    return showblks(filters, rcsize=rcsize, stride=stride,
                    plot=plot, bgcolor=bgcolor, title=title)
