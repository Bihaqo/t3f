# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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