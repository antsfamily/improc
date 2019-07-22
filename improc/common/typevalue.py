#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import numpy as np
from ..utils.log import *


def peakvalue(A, dtype=None):
    """Find peak value in matrix

    Find peak value in matrix

    Parameters
    ----------
    A : {numpy array}
        Data for finding peak value

    Returns
    -------
    number
        Peak value.
    """

    logging.info("---In peakvalue...")

    if dtype in ('float', 'float16', 'float32', 'float64', 'float128'):
        Vpeak = 1
    elif dtype in ('uint8', 'uint16', 'uint32', 'uint64'):
        datatype = str(dtype)
        Vpeak = 2 ** int(datatype[4:]) - 1
    elif dtype in ('int64', 'int32', 'int16', 'int8'):
        datatype = str(dtype)
        Vpeak = 2 ** int(datatype[3:]) / 2 - 1
    else:
        logging.info("~~~Unknown type, using the maximum value!")
        Vpeak = np.max(A)
    if dtype is None:
        logging.info("~~~Using the maximum value!")
        Vpeak = np.max(A)


    logging.info("---Out peakvalue.")
    return Vpeak
