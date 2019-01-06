Comparison to other libraries
=============================

A brief overview of other libraries that support Tensor Train decomposition (which is also known under the name Matrix Product State in physics community).

+---------------+-------------------+---------+------------+--------------+-----------+
| | Library     | | Language        | | GPU   | | autodiff | | Riemannian | | DMRG    |
| |             | |                 | |       | |          | |            | | AMen    |
| |             | |                 | |       | |          | |            | | TT-cross|
+===============+===================+=========+============+==============+===========+
| t3f           | Python/TensorFlow | Yes     | Yes        | Yes          | No        |
+---------------+-------------------+---------+------------+--------------+-----------+
| tntorch_      | Python/PyTorch    | Yes     | Yes        | No           | No        |
+---------------+-------------------+---------+------------+--------------+-----------+
| ttpy_         | Python            | No      | No         | Yes          | Yes       |
+---------------+-------------------+---------+------------+--------------+-----------+
| mpnum_        | Python            | No      | No         | No           | DMRG      |
+---------------+-------------------+---------+------------+--------------+-----------+
| `scikit_tt`_  | Python            | No      | No         | No           | No        |
+---------------+-------------------+---------+------------+--------------+-----------+
| mpys_         | Python            | No      | No         | No           | No        |
+---------------+-------------------+---------+------------+--------------+-----------+
| `TT-Toolbox`_ | Matlab            | Partial | No         | No           | Yes       |
+---------------+-------------------+---------+------------+--------------+-----------+
| TENSORBOX_    | Matlab            | Partial | No         | ??           | ??        |
+---------------+-------------------+---------+------------+--------------+-----------+
| Tensorlab_    | Matlab            | Partial | No         | ??           | ??        |
+---------------+-------------------+---------+------------+--------------+-----------+
| ITensor_      | C++               | No      | No         | No           | DMRG      |
+---------------+-------------------+---------+------------+--------------+-----------+
| libtt_        | C++               | No      | No         | No           | TT-cross  |
+---------------+-------------------+---------+------------+--------------+-----------+


.. _tntorch: https://github.com/rballester/tntorch
.. _ttpy: https://github.com/oseledets/ttpy
.. _mpnum: https://github.com/dseuss/mpnum
.. _scikit\_tt: https://github.com/PGelss/scikit_tt
.. _mpys: https://github.com/alvarorga/mpys
.. _TT-Toolbox: https://github.com/oseledets/TT-Toolbox
.. _TENSORBOX: http://www.bsp.brain.riken.jp/~phan/#tensorbox
.. _Tensorlab: https://www.tensorlab.net
.. _ITensor: http://itensor.org/
.. _libtt: https://bitbucket.org/matseralex/tt_smoluh/src/master/libtt/

If you use python, we would suggest using t3f if you need extensive Riemannian optimization support, t3f or tntorch if you need GPU or autodiff support, and ttpy if you need advanced algorithms such as AMen.

The performance of the libraries is a bit tricky to measure fairly and is actually not that different between the libraries because everyone relies on the same BLAS/MKL subruitines. However, GPU can help a lot if you need operations that can be expressed as large matrix-by-matrix multiplications, e.g. computing a gram matrix of a bunch of tensors. For more details on benchmarking t3f see :doc:`benchmarking`.
