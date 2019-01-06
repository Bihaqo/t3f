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
If you use T3F in your research work, we kindly ask you to cite [the paper](https://arxiv.org/abs/1801.01928) describing this library
```
@article{novikov2018tensor,
  title={Tensor Train decomposition on TensorFlow (T3F)},
  author={Novikov, Alexander and Izmailov, Pavel and Khrulkov, Valentin and Figurnov, Michael and Oseledets, Ivan},
  journal={arXiv preprint arXiv:1801.01928},
  year={2018}
}
```