#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-14 16:14:12
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

from libtiff import TIFF, TIFFfile
import tifffile


def tifsave(filepath, A, Info=None):

    # tif = TIFF.open(filepath, mode='w')
    # A = np.transpose(A, (2, 0, 1))

    # C, H, W = A.shape

    # A = A.copy(order='K')

    # tif.SetField('ImageDescription', Info['ImageDescription'])
    # tif.SetField('ImageWidth', Info['ImageWidth'])
    # tif.SetField('ImageLength', Info['ImageLength'])
    # tif.SetField('SamplesPerPixel', Info['SamplesPerPixel'])
    # tif.SetField('BitsPerSample', Info['BitsPerSample'])
    # tif.SetField('SampleFormat', Info['SampleFormat'])
    # tif.SetField('Compression', Info['Compression'])
    # tif.SetField('RowsPerStrip', Info['RowsPerStrip'])
    # tif.SetField('StripByteCounts', Info['StripByteCounts'].value)
    # tif.SetField('XResolution', Info['XResolution'])
    # tif.SetField('XResolution', 1)
    # tif.SetField('YResolution', Info['YResolution'])
    # tif.SetField('YResolution', 1)
    # tif.SetField('Orientation', 1)
    # tif.SetField('Orientation', Info['Orientation'])

    # if Info is not None:
    #     for k, v in Info.items():
    #         print(k, v)
    #         if k is 'StripByteCounts':
    #             tif.SetField(k, v.value)
    #         else:
    #             if v is not None:
    #                 tif.SetField(k, v)

    # tif.write_image(A, compression='jpeg', write_rgb=False)
    # tif.close()

    tifffile.imsave(filepath, A)

    return 0


def tifinfo(filepath):
    """Get tif file information

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

    tif = TIFF.open(filepath, mode='r')

    Info = dict()

    Info['ImageDescription'] = tif.GetField('ImageDescription')
    Info['ImageWidth'] = tif.GetField('ImageWidth')
    Info['ImageLength'] = tif.GetField('ImageLength')
    Info['SamplesPerPixel'] = tif.GetField('SamplesPerPixel')
    Info['BitsPerSample'] = tif.GetField('BitsPerSample')
    Info['SampleFormat'] = tif.GetField('SampleFormat')
    Info['Compression'] = tif.GetField('Compression')
    Info['RowsPerStrip'] = tif.GetField('RowsPerStrip')
    Info['StripByteCounts'] = tif.GetField('StripByteCounts')
    Info['XResolution'] = tif.GetField('XResolution')
    Info['YResolution'] = tif.GetField('YResolution')
    Info['Orientation'] = tif.GetField('Orientation')
    # Info['PhotometricInterpretation'] = tif.GetField('PhotometricInterpretation')
    # Info['PlanarConfiguration'] = tif.GetField('PlanarConfiguration')

    tif.close()

    print("ImageDescription: ", Info['ImageDescription'])
    print("ImageWidth: ", Info['ImageWidth'])
    print("ImageLength: ", Info['ImageLength'])
    print("SamplesPerPixel: ", Info['SamplesPerPixel'])
    print("BitsPerSample: ", Info['BitsPerSample'])
    print("SampleFormat: ", Info['SampleFormat'])
    print("Compression: ", Info['Compression'])
    print("RowsPerStrip: ", Info['RowsPerStrip'])
    print("StripByteCounts: ", Info['StripByteCounts'])
    print("XResolution: ", Info['XResolution'])
    print("YResolution: ", Info['YResolution'])
    print("Orientation: ", Info['Orientation'])
    # print("PhotometricInterpretation: ", Info['PhotometricInterpretation'])
    # print("PlanarConfiguration: ", Info['PlanarConfiguration'])

    return Info


def tifread(filepath, verbose=False):

    if verbose:
        Info = tifinfo(filepath)

    tif = TIFF.open(filepath, mode='r')
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
    import tifffile

    # inimgfile = '/mnt/d/DataSets/hsi/RemoteSensing/HSI/image1.tif'
    # outimgfile = '/mnt/d/DataSets/hsi/RemoteSensing/HSI/image1out.tif'

    inimgfile = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/train/train/img_2018/image_2018_960_960_1.tif'
    outimgfile = '/mnt/d/DataSets/oi/rsi/RSSRAI2019/train/train/img_2018/image_2018_960_960_1OO.tif'

    # Iin = tifffile.imread(inimgfile)
    # tifffile.imsave(outimgfile, Iin)
    # Iout = tifffile.imread(outimgfile)

    Iin, Info = tifread(inimgfile, verbose=True)

    print("Iin.shape: ", Iin.shape)
    print("min(Iin), min(Iin): ", np.min(Iin), np.max(Iin))

    tifwrite(Iin, outimgfile, Info=Info)
    Iout, Info = tifread(outimgfile, verbose=True)

    print("Iout.shape: ", Iout.shape)
    print("min(Iout), min(Iout): ", np.min(Iout), np.max(Iout))

    plt.figure()
    plt.subplot(121)
    plt.imshow(Iin[:, :, 1])
    plt.subplot(122)
    plt.imshow(Iout[:, :, 1])
    plt.tight_layout()
    plt.show()
