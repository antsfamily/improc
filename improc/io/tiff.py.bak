#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-14 16:14:12
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import libtiff
import tifffile


def tifwrite(filepath, A, Info=None):
    r"""save array to tif file

    save image array to tif file

    Parameters
    ----------
    filepath : {str}
        file path for saving data
    A : {numpy array}
        data (:math:`H×W×C`) to be saved
    Info : {information}, optional
        image information (the default is None)

    Returns
    -------
    number
        if 0, successed
    """
    if np.ndim(A) > 2:
        H, W, C = A.shape
    else:
        H, W = A.shape
    Info['ResolutionUnit'] = 'Inch'

    with tifffile.TiffWriter(filepath) as tif:
        tif.save(A, compress=0, photometric='minisblack')

    # self, data=None, shape=None, dtype=None, returnoffset=False,
    #              photometric=None, planarconfig=None, extrasamples=None, tile=None,
    #              contiguous=True, align=16, truncate=False, compress=0,
    #              rowsperstrip=None, predictor=False, subsampling=None,
    #              colormap=None, description=None, datetime=None, resolution=None,
    #              subfiletype=0, software='tifffile.py', metadata={},
    #              ijmetadata=None, extratags=()
    return 0


def tifsave(filepath, A, Info=None):
    r"""save array to tif file

    save image array to tif file

    Parameters
    ----------
    filepath : {str}
        file path for saving data
    A : {numpy array}
        data (:math:`H×W×C`) to be saved
    Info : {information}, optional
        image information (the default is None)

    Returns
    -------
    number
        if 0, successed
    """

    with tifffile.TiffWriter(filepath) as tif:
        # print(tif.sampleformat)
        tif.save(A,
                 photometric=TIFF.PHOTOMETRIC.MINISBLACK,
                 planarconfig=TIFF.PLANARCONFIG.CONTIG,
                 compress=0,
                 resolution=(None, None, TIFF.RESUNIT.INCH),
                 rowsperstrip=8,
                 # orientation=TIFF.ORIENTATION.TOPLEFT,
                 contiguous=False)

    tifffile.imsave(filepath, A)

    return 0


def tifinfo(filepath):
    r"""Get tif file information

    Get tif file information

    ImageDescription
    ImageWidth
    ImageLength
    SamplesPerPixel
    SampleFormat
    Compression
    RowsPerStrip
    XResolution
    YResolution
    Orientation
    PhotometricInterpretation
    PlanarConfiguration

    Parameters
    ----------
    filepath : {string}
        tif file path.

    Returns
    -------
    dict
        Information dict
    """

    # tif = TIFF.open(filepath, mode='r')

    Info = dict()

    # Info['ImageDescription'] = tif.GetField('ImageDescription')
    # Info['ImageWidth'] = tif.GetField('ImageWidth')
    # Info['ImageLength'] = tif.GetField('ImageLength')
    # Info['SamplesPerPixel'] = tif.GetField('SamplesPerPixel')
    # Info['BitsPerSample'] = tif.GetField('BitsPerSample')
    # Info['SampleFormat'] = tif.GetField('SampleFormat')
    # Info['Compression'] = tif.GetField('Compression')
    # Info['RowsPerStrip'] = tif.GetField('RowsPerStrip')
    # Info['StripByteCounts'] = tif.GetField('StripByteCounts')
    # Info['XResolution'] = tif.GetField('XResolution')
    # Info['YResolution'] = tif.GetField('YResolution')
    # Info['Orientation'] = tif.GetField('Orientation')
    # # Info['PhotometricInterpretation'] = tif.GetField('PhotometricInterpretation')
    # # Info['PlanarConfiguration'] = tif.GetField('PlanarConfiguration')

    # tif.close()


    with tifffile.TiffFile(filepath) as tif:
        # image_stack = tif.asarray()
        for page in tif.pages:
            for tag in page.tags.values():
                tag_name, tag_value = tag.name, tag.value
                Info[tag_name] = tag_value

    return Info


def tifread(filepath, verbose=False):
    r"""read data from tif file

    read data from tif file

    Parameters
    ----------
    filepath : {string}
        tif file path
    verbose : {bool}, optional
        show more image information (the default is False)

    Returns
    -------
    numpy array
        data array :math:`H×W×C`
    """

    if verbose:
        Info = tifinfo(filepath)

    tif = libtiff.TIFF.open(filepath, mode='r')
    A = tif.read_image()
    tif.close()

    # A = tifffile.imread(filepath)

    if verbose:
        return A, Info
    else:
        return A


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    inimgfile = '/mnt/d/DataSets/hsi/RemoteSensing/HSI/image1.tif'
    outimgfile = '/mnt/d/DataSets/hsi/RemoteSensing/HSI/image1out.tif'

    # inimgfile = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/new/train/train/img_2018/image_2018_960_960_1.tif'
    # outimgfile = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/new/train/train/img_2018/image_2018_960_960_1a.tif'


    # Iin = tifffile.imread(inimgfile)
    # tifffile.imsave(outimgfile, Iin)
    # Iout = tifffile.imread(outimgfile)

    Iin, Info = tifread(inimgfile, verbose=True)
    # Iin = tifread(inimgfile, verbose=False)
    # Iin = np.transpose(Iin, (2, 0, 1))

    print("Iin.shape: ", Iin.shape)
    print("min(Iin), min(Iin): ", np.min(Iin), np.max(Iin))

    tifwrite(outimgfile, Iin, Info=Info)
    Iout, Info = tifread(outimgfile, verbose=True)
    # Iout = tifread(outimgfile, verbose=False)

    print("Iout.shape: ", Iout.shape)
    print("min(Iout), min(Iout): ", np.min(Iout), np.max(Iout))

    plt.figure()
    plt.subplot(121)
    plt.imshow(Iin[:, :])
    plt.subplot(122)
    plt.imshow(Iout[:, :])
    plt.tight_layout()
    plt.show()
