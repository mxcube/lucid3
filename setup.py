#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = ['scipy', 'numpy', 'opencv-python>=2.4', 'matplotlib']

setup(name='lucid3',
      version='1.0',
      description='ESRF Target Detection Software',
      author='Sogeti High Tech',
      packages=['lucid3'],
      install_requires=requirements,
      python_requires='>=2.7',
      )
