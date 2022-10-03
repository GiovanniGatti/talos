#!/usr/bin/env python
from setuptools import setup

try:
    import tensorflow
except ImportError as e:
    raise ModuleNotFoundError('Expected tensorflow>=2.2.0 to be provided')

setup(name='talos',
      version='0.1.0-SNAPSHOT',
      author='Giovanni Gatti Pinheiro',
      author_email='giovanni.gattipinheiro@amadeus.com',
      setup_requires=['wheel'],
      package_dir={'': 'src'},
      packages=['talos'],
      package_data={'talos': ['py.typed']},
      install_requires=['ray[rllib]==1.7.0', 'statsmodels==0.13.0', 'gym==0.21.0', 'matplotlib==3.5.3'],
      python_requires='>=3.7')
