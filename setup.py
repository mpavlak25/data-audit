#!/usr/bin/env python

from setuptools import setup

__author__ = 'JHU'
__version__ = '0.1'

setup(
    name='dm-audit',
    version=__version__,
    description='Targeted Auditing for Medical Image Datasets',
    long_description=open('README.md').read(),
    author=__author__,
    author_email='',
    license='BSD',
    packages=['dmaudit'],
    keywords='bias, deep learning',
    classifiers=[],
    install_requires=[]
)
