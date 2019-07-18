#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-14 16:14:12
# @Author  : Yan Liu & Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.0$

import os
import h5py
import sys
import pickle as pkl
import scipy.io as scio
import numpy as np


def save(data, file):
    """save data to file

    save data to '.pkl' or '.mat' file

    Parameters
    ----------
    data : {dict}
        data dict to be saved
    file : {string}
        specify where to save
    """

    ext = os.path.splitext(file)[1]
    if ext == '.pkl':
        f = open(file, 'wb')
        pkl.dump(data, f, 0)
        f.close()
    elif ext == '.mat':
        scio.savemat(file, {'data': data})
    return 0

def load(file):
    """load data from file

    load data from '.pkl' or '.mat' file

    Parameters
    ----------
    file : {string}
        specify which file to load
    """

    filename, EXT = os.path.splitext(file)

    if EXT == '.pkl':
        f = open(file, 'rb')
        # for python2
        if sys.version_info < (3, 1):
            data = pkl.load(f)
        # for python3
        else:
            data = pkl.load(f, encoding='latin1')
        f.close()

    elif EXT == '.mat':
        data = scio.loadmat(file, struct_as_record=True)
    return data