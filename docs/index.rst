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
    faq
    api
    comparison
    benchmark
    troubleshooting

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorials/tensor_nets
    tutorials/tensor_completion
    tutorials/riemannian

Citation
--------

If you use T3F in your research work, we kindly ask you to cite the paper_ describing this library

.. _paper: http://jmlr.org/papers/v21/18-008.html

.. code-block:: console

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
