#!/usr/bin/env python2

from distutils.core import setup

setup(name='plotextract',
      version='0.0',
      description='Extract digital series data from raster images',
      author='niemandkun',
      author_email='niemandkun@yandex.ru',
      url='none',
      packages=['plotvision'],
      scripts=['bin/plotextract.py'],
      install_requires=[
        'numpy',
        'matplotlib',
        'scikit-image',
        'scikit-learn',
        'pillow',
        'pytesseract',
        'scipy',
        'SimpleITK',
        'pandas',
      ],
     )
