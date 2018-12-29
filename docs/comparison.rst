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
| ITensor_      | C++               | No      | No         | No           | DMRG      |
+---------------+-------------------+---------+------------+--------------+-----------+


.. _tntorch: https://github.com/rballester/tntorch
.. _ttpy: https://github.com/oseledets/ttpy
.. _mpnum: https://github.com/dseuss/mpnum
.. _scikit\_tt: https://github.com/PGelss/scikit_tt
.. _mpys: https://github.com/alvarorga/mpys
.. _TT-Toolbox: https://github.com/oseledets/TT-Toolbox
.. _ITensor: http://itensor.org/