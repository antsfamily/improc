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
from ..utils.preprocessing import scalearr, imgdtype
from ..utils.log import *


r"""
Functions to split some images into blocks.

Functions can split different size images randomly or orderly(column-wise or
row-wise):

.. autosummary::
    :nosignatures:

    imgs2blks
    blks2imgs

"""

# np.random.seed(2016)


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
                    #   plt.figure(cnt)
                    #   plt.imshow(blks[:,:,:,cnt])
                    cnt = cnt + 1
    return blks


def imgs2blks(imgs, blksize=[8, 8, 3], padmode='symmetric'):
    r"""
    blks, imgsShape = imgs2blks(imgs, blksize) trys to split the imgs(an H-W-C
    -N numpy ndarray, or list of images filepathes) into image blocks(an bH-bW-
    bC-bN ndarray) orderly; imgsShape contains each image's size in imgs for
    fighting back to images.

    Trys to split some images into blocks orderly (1.channel-wisely, 2.colum-
    wisely, 3.row-wisely).

    Parameters
    ----------
    imgs : {array_like, or list of image pathes}.
        Images to be splited, a H-W-C-N numpy ndarray, or a list of image
        pathes with diffrent image shapes.
    blksize : {int tuple, list or None, optional}
        Specifies the each block size (rows, cols, channel) that you want to
        split. If not given, blksize=[8, 8, 3].
    padmode : {str or None, optional}
        Padding mode when an image can't be split fully. 'symmetric' or 'edge'
        can be choosen, see np.pad for more option. The Channel will also be
        pad if it is not enough, e.g., for an H-W-1(gray) image, if you want
        to get blksize=[8,8,3].

    Returns
    -------
    blks : {ndarray}
        A bH-bW-bC-bN numpy ndarray, (bH, bW, bC) == blksize.
    imgsshape : {tuple}
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
    logging.info("---In imgs2blks...")

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
        logging.info("---Out imgs2blks...")

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
            logging.info("---Out imgs2blks...")
            return blks[:, :, :, 1:], imgsShape  # blks[:,:,:,0] all zeros
        else:
            raise TypeError('No avaliable image!')
        # return patches
    else:
        raise TypeError(
            '"imgs" should be a path list or H-W-C-N numpy ndarray!')
    logging.info("---Out imgs2blks.")


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
    r"""
    Fight image blocks back to images.

    Parameters
    ----------
    blks : {array_like}
        A bH-bW-bC-bN numpy ndarray, where, bH, bW, bC, bN specify height,
        width, channels, number of images respectivelly. If blks Contains
        Gray Images and RGB Images, then bC = 3, and Gray Images is copied.
    imgsShape : {tuple or list}
        Specify each image's size.
    index : {int, optional}
        The image that you want to fight back.
    tofolder : {str (path), optional}
        Save specified images into files: image_0.png, image_i.png, ... ,
        image_N.png

    Returns
    -------
    out : {ndarray or bool}
        A list of image numpy ndarray(H-W-C).

    See Also
    --------
    imgs2ptcs, imgs2blks.

    """

    logging.info("---In blks2imgs...")

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
            imwriteadv(tofolder + '/image_' + str(n) + '.png', img)

    logging.info("---Out blks2imgs.")

    return imgs  # list contains N images numpy ndarray
