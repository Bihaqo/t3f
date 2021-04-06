[![Build Status](https://travis-ci.org/Bihaqo/t3f.svg?branch=develop)](https://travis-ci.org/Bihaqo/t3f)
[![Coverage Status](https://coveralls.io/repos/github/Bihaqo/t3f/badge.svg?branch=develop)](https://coveralls.io/github/Bihaqo/t3f?branch=develop)

TensorFlow implementation of a library for working with Tensor Train (TT) decomposition which is also known as Matrix Product State (MPS).

# Documentation
The documentation is available via [readthedocs](https://t3f.readthedocs.io/en/latest/index.html).

# Comparison with other libraries
There are about a dozen other libraries implementing Tensor Train decomposition. 
The main difference between `t3f` and other libraries is that `t3f` has extensive support for Riemannian optimization and that it uses TensorFlow as backend and thus supports GPUs, automatic differentiation, and batch processing. For a more detailed comparison with other libraries, see the [corresponding page](https://t3f.readthedocs.io/en/latest/comparison.html) in the docs.

# Tests
```bash
nosetests  --logging-level=WARNING
```

# Building documentation
The documentation is build by sphinx and hosted on readthedocs.org. To locally rebuild the documentation, install sphinx and compile the docs by
```bash
cd docs
make html
```

# Citing
If you use T3F in your research work, we kindly ask you to cite [the paper](http://jmlr.org/papers/v21/18-008.html) describing this library
```

@article{JMLR:v21:18-008,
  author  = {Alexander Novikov and Pavel Izmailov and Valentin Khrulkov and Michael Figurnov and Ivan Oseledets},
  title   = {Tensor Train Decomposition on TensorFlow (T3F)},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {30},
  pages   = {1-7},
  url     = {http://jmlr.org/papers/v21/18-008.html}
}
```