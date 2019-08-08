#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2015-10-15 10:34:16
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$


import numpy as np


def randperm(start, end, number):
    """randperm function like matlab

    genarates diffrent random interges in range [start, end)

    Parameters
    ----------
    start : {integer}
        start sampling point

    end : {integer}
        end sampling point
    number : {integer}
        random numbers
    """

    P = np.random.permutation(range(start, end))

    return P[0:number]


def randperm2d(H, W, number, population=None):
    """randperm 2d function

    genarates diffrent random interges in range [start, end)

    Parameters
    ----------
    H : {integer}
        height

    W : {integer}
        width
    number : {integer}
        random numbers
    population : {list or numpy array(1d or 2d)}
        part of population in range(0, H*W)
    """

    if population is None:
        population = range(0, H * W)

    population = np.array(population).flatten()
    population = np.random.permutation(population)

    Ph = np.floor(population / W).astype('int')
    Pw = np.floor(population - Ph * W).astype('int')

    # print(Pw + Ph * W)
    return Ph, Pw


if __name__ == '__main__':

    R = randperm(2, 10, 8)
    print(R)

    Rh, Rw = randperm2d(5, 6, 8)

    print(Rh)
    print(Rw)
