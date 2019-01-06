[![Build Status](https://travis-ci.org/Bihaqo/t3f.svg?branch=develop)](https://travis-ci.org/Bihaqo/t3f)
[![Coverage Status](https://coveralls.io/repos/github/Bihaqo/t3f/badge.svg?branch=develop)](https://coveralls.io/github/Bihaqo/t3f?branch=develop)

TensorFlow implementation of a library for working with Tensor Train (TT) decomposition which is also known as Matrix Product State (MPS).

# Documentation
The documentation is available via [readthedocs](https://t3f.readthedocs.io/en/latest/index.html).

# Comparison with other libraries
There are implementations of the TT-toolbox in [plain Python](https://github.com/oseledets/ttpy) and [Matlab](https://github.com/oseledets/TT-Toolbox). Also, there is a very nice generic tensor network library [tnttorch](https://github.com/rballester/tntorch) which supports TT as a special case.

The main difference between `t3f` Python/Matlab implementations is that `t3f` uses TensorFlow as backend and thus supports GPUs, automatic differentiation, and batch processing. As a more generic library, `tnttorch` lacks some TensorTrain specific tools (e.g. Riemannian optimization support).

Here are the results im ms of benchmarking T3F on CPU and GPU and comparing against the [TTPY library](https://github.com/oseledets/ttpy)
<img src="examples/profile/results.png" height="200">

For more details see ```examples/profile``` folder.

# Tests
```bash
nosetests  --logging-level=WARNING
```

# Building API documentation
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