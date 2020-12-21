from distutils.core import setup

setup(name='genSTEM',
      version='0.1',
      packages=['genSTEM'],
      install_requires = [
          'numpy',
          #'cupy',
          'sympy',
          'ase'
      ]
      )