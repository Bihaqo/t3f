.. t3f documentation master file, created by
   sphinx-quickstart on Sun Mar 12 10:06:09 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quick start
===========

Import the libraries

.. code-block:: python

	import tensorflow as tf
	import t3f

Generate a random tensor and compute its norm.

.. code-block:: python

	# Create a random tensor of shape (3, 2, 2).
	a = t3f.random_tensor((3, 2, 2), tt_rank=3)
	norm = t3f.frobenius_norm(a)
	# Convert TT-tensor into a dense tensor for printing.
	a_full = t3f.full(a)
	# Run a tensorflow session to run the operations.
	with tf.Session() as sess:
	  # Run the operations. Note that if you run these
	  # two operations separetly (sess.run(a_full), sess.run(norm))
	  # the result will be different, since sess.run will
	  # generate a new random tensor a on each run because `a' is
	  # an operation 'generate me a random tensor'.
	  a_val, norm_val = sess.run([a_full, norm])
	  print('The norm is %f' % norm_val)
	  print(a_val)

Arithmetic operations

.. code-block:: python

	a = t3f.random_tensor((3, 2, 2), tt_rank=3)
	b_dense = tf.random_normal((3, 2, 2))
	# Use TT-SVD on b_dense.
	b_tt = t3f.to_tt_tensor(b_dense, max_tt_rank=4)
	sum_round = t3f.round(t3f.add(a, b_tt), max_tt_rank=2)

Tensor operations

.. code-block:: python

	# Inner product (sum of products of all elements).
	a = t3f.random_tensor((3, 2, 2), tt_rank=3)
	b = t3f.random_tensor((3, 2, 2), tt_rank=4)
	inner_prod = t3f.flat_inner(a, b)

Matrix operations

.. code-block:: python

	A = t3f.random_matrix(((3, 2, 2), (2, 3, 3)), tt_rank=3)
	b = t3f.random_matrix(((2, 3, 3), None), tt_rank=3)
	# Matrix-by-vector
	matvec = t3f.matmul(A, b)

	# Matrix-by-dense matrix
	b_dense = tf.random_normal((18, 1))
	matvec2 = t3f.matmul(A, b_dense)
