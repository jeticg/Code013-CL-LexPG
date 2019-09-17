# -*- coding: utf-8 -*-
import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(["*coder/*.pyx",
                           "*.pyx",
                           "decoder/plugins/*.pyx"])
)

setuptools.setup(
    name="CL-LexPG",
    author="Jetic Gū",
    install_requires=[
        'Cython',
        'ConfigParser',
        'six',
        'natlang',
    ],
)
