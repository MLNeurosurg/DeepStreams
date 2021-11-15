from setuptools import setup, find_packages

setup(name='deepstreams',
      version='0.1',
      packages=find_packages(),
      install_requires=[
          'setuptools>=41.0.0', 'torchvision>=0.6.1', 'torch>=1.5.0',
          'imageio'
      ])