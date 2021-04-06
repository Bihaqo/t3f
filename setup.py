from setuptools import setup

setup(name='t3f',
      version='1.2.0',
      description='Tensor Train decomposition on TensorFlow',
      url='https://github.com/Bihaqo/t3f',
      author='Alexander Novikov',
      author_email='sasha.v.novikov@gmail.com',
      license='MIT',
      packages=['t3f'],
      install_requires=[
            'numpy',
      ],
      zip_safe=False)
