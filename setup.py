from setuptools import setup

setup(name='ddi',
      version='0.0.1',
      description='',
      url='https://github.com/CMI-UZH/side-effects',
      packages=['ddi'],
      python_requires='>=3.6.0',
      install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'scikit-learn',
            'torch',
            'matplotlib',
            'seaborn'
      ],
      zip_safe=False)