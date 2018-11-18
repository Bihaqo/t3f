t3f - library for working with Tensor Train decomposition built on top of TensorFlow
=============================

t3f is a library for working with Tensor Train decomposition. Tensor Train decomposition is a generalization of the low-rank decomposition from matrices to tensors (=multidimensional arrays), i.e. it's a tool to efficiently work with structured tensors.
t3f is implemented on top of TensorFlow which gives it a few nice properties:

- GPU support -- just run your model on a machine with a CUDA-enabled GPU and GPU version of the TensorFlow, and t3f will execute most of the operations on it.
- Autodiff -- TensorFlow can automatically compute the derivative of a function with respect to the underlying parameters of the Tensor Train decomposition (TT-cores). Also, if you are into the Riemannian optimization, you can automatically compute the Riemannian gradient of a given function. Don't worry if you don't know what it is :)
- Batch processing -- you can run a single vectorized operation on a set of Tensor Train objects.
- Easy to use with Deep Learning, e.g. you can define a layer parametrized with a Tensor Train object and use it as a part of your favorite neural network implemented in TensorFlow.

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    quick_start
    api

Citation
--------

If you use T3F in your research work, we kindly ask you to cite the paper_ describing this library

.. _paper: https://arxiv.org/abs/1801.01928

.. code-block:: console

    @article{novikov2018tensor,
	  title={Tensor Train decomposition on TensorFlow (T3F)},
	  author={Novikov, Alexander and Izmailov, Pavel and Khrulkov, Valentin and Figurnov, Michael and Oseledets, Ivan},
	  journal={arXiv preprint arXiv:1801.01928},
	  year={2018}
	}
