# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

## [1.0.0] - 2018-01-08
### Added
- API reference documentation.
- Coveralls test coverage.
- add_projected function for adding tensors from the same tangent space.
- add_n_projected function.
- Test TF 1.0-1.4
- Test Python 2.7, 3.4, 3.5, 3.6
- tf.kronecker module for working with Kornecker product matrices (TT-matrices of TT-rank 1).
- Example Jupyter notebooks with basic functionality, TensorNet (training neural networks with params in the TT-format), Riemannian optimization, and tensor completion.
- A profiler for basic operations on CPU and GPU.
- approximate.add_n function for adding together a batch of tensors and rounding after each summation.
- Manually import all the relevant function in __init__ and avoid importing internal tools.
- gather_nd for gathering several elements from a TT-objects at once.
- Overload operations to support `a - b`, `a + -b`, `0.4 * a`.
- Add complexities of some functions into the docstrings.
- Glorot, He, LeCun initializers for TT-matrices

### Changed
- Speed improvements (in particular to quadratic_form).
- Bug fixes.
- Multiplication by a number uniformly multiplies all cores (improved stability for large tensors).
- Better test coverage.

## [0.3.0] - 2017-04-20
### Added
- Python 3 support.
- Speed improvements.
- Bug fixes.
- Travis CI.
- project_matmul -- fast projection of matrix-by-vector product.
- pairwise_flat_inner_projected -- fast pairwise dot products between projections on the same tangent space.
- multiply_along_batch_dim
- Support for indexing with placeholders or tf.Tensors.
- Support for t3f.cast(batch_tt).
- A function to replace tf.svd by np.svd.

### Changed
- Better Frobenius norm implementation (via QR).
- Riemannian projection API: now it's project(what, where). 


## [0.2.0] - 2017-03-23
### Added
- (Partial) support for batches of TT-tensors.
- Riemannian module (projection on the tangent space).
- op property and str method for TensorTrain
- concat_along_batch_dim
- expand_batch_dim
- gram_matrix
- Multiplication by a number

### Changed
- Fix add function for dtypes not equal tf.float32
- flat_inner and quadratic_form now return numbers (instead of 1 x 1 tensors)

## [0.1.0] - 2017-03-12
### Added
- Indexing (e.g. TensorTrain[:, 3, 2:4])
- Full (converting TT to dense)
- TT-SVD and rounding
- Basic arithmetic (add, multiply, matmul, flat_inner)
- Variables support
- Kronecker module (functions for TT-rank 1 TT-matrices)
- quadratic_form
- frobenius_norm

[Unreleased]: https://github.com/Bihaqo/t3f/compare/master...develop
[0.2.0]: https://github.com/Bihaqo/t3f/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/Bihaqo/t3f/compare/f24409508...0.1.0
