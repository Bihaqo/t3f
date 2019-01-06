Benchmark
================

The performance of different libraries implementing Tensor Train decomposition is a bit tricky to compare fairly and is actually not that different because everyone relies on the same BLAS/MKL subruitines.

So the main purpose of this section is not to prove that T3F is faster than every other library (it is not), but rather to assess GPU gains on different ops and identify bottlenecks by comparing to *some* other library. As a reference implementation, we decided to use ttpy_ library.

.. _ttpy: https://github.com/oseledets/ttpy

See the following table for time in ms of different opeartions run in ttpy_ (second column) and in t3f (other columns).

===============  ==================  ============  ============  ==============  ==============
Operation        ttpy, one on CPU    one on CPU    one on GPU    batch on CPU    batch on GPU
===============  ==================  ============  ============  ==============  ==============
matvec                       11.142         1.19          0.744           1.885           0.14
matmul                       86.191         9.849         0.95           17.483           1.461
norm                          3.79          2.136         1.019           0.253           0.044
round                        73.027        86.04        165.969           8.234         161.102
gram                          0.145         0.606         0.973           0.021           0.001
project_rank100             116.868         3.001        13.239           1.645           0.226
===============  ==================  ============  ============  ==============  ==============

The timing in the "batch" columns represent running the operation for a 100 of objects at the same time and then reporting the time per object. E.g. the last number in the first row (0.14) means that multiplying a single TT-matrix by 100 different TT-vectors takes 14 ms on GPU when using T3F, which translates to 0.14 ms per vector.

Note that rounding operation is slow on GPU. This is a `known TensorFlow bug`_, that the SVD implementation is slower on GPU than on CPU.

.. _`known TensorFlow bug`: https://github.com/tensorflow/tensorflow/issues/13603

The benchmark was run on NVIDIA DGX-1 server with Tesla V100 GPU and Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz with 80 logical cores

To run this benchmark on your own hardware, see `docs/benchmark`_ folder.

.. _`docs/benchmark`: https://github.com/Bihaqo/t3f/tree/develop/docs/benchmark
