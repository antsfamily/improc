#!/usr/bin/env python
# Copyright (c) 2015-2017, Zhi Liu.  All rights reserved.

from setuptools import setup
from setuptools import find_packages


setup(name='improc',
      version='1.0.0',
      description="A simple image process tool, \
                    especially for block or patch operation",
      author='Zhi Liu',
      author_email='zhiliu.mind@gmail.com',
      url='http://blog.csdn.net/enjoyyl',
      download_url='https://github.com/',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib'
      ],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot>=1.2.0'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      keywords=[
          'improc',
          'Image Processing',
          'Machine Learning',
          'AI'
      ]
      )
