#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import typing
import numpy as np
from ..utils.log import *
from numbers import Real
from typing import Dict, Tuple


TypesRangeDict = {
    "uint8": (0, 255),
    "uint16": (0, 65535),
    "float": (0.0, 1.0),
    "float16": (0.0, 1.0),
    "float32": (0.0, 1.0),
    "float64": (0.0, 1.0),
}


def get_drange(dtype):
    """This also assumes that image are considered where by convention
        for floats values are stored within range 0.0 to 1.0."""
    drange = TypesRangeDict[dtype.name]
    if drange is None:
        raise TypeError("{dtype} is a type that cannot be handled.")
    return drange


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
