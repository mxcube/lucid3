#!/usr/bin/env python
# 	-*- coding: utf-8 -*-

#from distutils.core import setup
from setuptools import setup

requirements = ['setuptools', 'python']

setup(name='lucid3',
        version='1.0',
        description='ESRF Target Detection Software',
        author='Sogeti High Tech',
        packages=['lucid3'],
        install_requires=requirements,
        python_requires='>=2.7',
     )
