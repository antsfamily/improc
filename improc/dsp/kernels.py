
import numpy as np

from scipy import ndimage


BOX_BLUR_3X3 = np.full((3, 3), 1.0 / 9.0,
                       dtype=float)
BOX_BLUR_5X5 = np.full((5, 5), 1.0 / 25.0,
                       dtype=float)
GAUSSIAN_BLUR_3x3 = (1.0 / 16.0) * np.array(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
VERTICAL_SOBEL_3x3 = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    dtype=float)
HORIZONTAL_SOBEL_3x3 = np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    dtype=float)

KER_DICT = dict()

KER_DICT['blur3x3_mean1'] = np.full((3, 3), 1.0 / 9.0, dtype=float)
KER_DICT['sharpen3x3_type1'] = np.array([[-1., -1., -1.], [-1., 9., -1.], [-1., -1., -1.]]) / 9.


def convolve(image, kernel):
    r"""Performs the convolution using provided kernel.


    Parameters
    ----------
    image : {2d or 3d array}
        image to be convolved :math:`H\times W \times C`. In case when image has multiple channels, the kernel is going to be used for each image channel.
    kernel : {2d array}
        kernel used for convolution

    Returns
    -------
    ndarray
        Convolved image (probably of different dtype)

    Raises
    ------
    ValueError
        - Kernel Bigger Than Image Error
        - Kernel Shape Not Odd Error
        - Kernel Not2 DArray
    """

    if kernel.ndim != 2:
        raise ValueError('Kernel is not an 2d-array, it a' + str(kernel.ndim) + '-d array.')
    if kernel.shape[:2] > image.shape[:2]:
        raise ValueError('Kernel Shape ' + str(kernel.shape) + ' is bigger then image shape' + str(image.shape))
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError('Kernel Shape ' + str(kernel.shape) + 'Not Odd')
    output = np.zeros(image.shape, dtype=float)
    if image.ndim > 2:
        for channel in range(image.shape[2]):
            output[:, :, channel] = (
                ndimage.convolve(image[:, :, channel],
                                 kernel,
                                 output=float))
    else:
        output = ndimage.convolve(image,
                                  kernel,
                                  output=float)

    output = output[kernel.shape[0] // 2: -(kernel.shape[0] // 2),
                    kernel.shape[1] // 2: -(kernel.shape[1] // 2)]
    return output


if __name__ == '__main__':

    print(BOX_BLUR_3X3)
    print(BOX_BLUR_5X5)
    print(GAUSSIAN_BLUR_3x3)
    print(VERTICAL_SOBEL_3x3)
    print(HORIZONTAL_SOBEL_3x3)
    print(KER_DICT)
