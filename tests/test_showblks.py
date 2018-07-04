#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-07-04 09:43:51
# @Author  : Zhi Liu (zhiliu.mind@gmail.com)
# @Link    : http://iridescent.ink
# @Version : $1.1$

import numpy as np
from improc.blkptcs import showblks


blks = np.uint8(np.random.randint(0, 255, (8, 8, 3, 101)))
showblks(blks, bgcolor='k')
