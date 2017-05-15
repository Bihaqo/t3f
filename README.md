[![Build Status](https://travis-ci.org/Bihaqo/t3f.svg?branch=develop)](https://travis-ci.org/Bihaqo/t3f)
[![Coverage Status](https://coveralls.io/repos/github/Bihaqo/t3f/badge.svg?branch=develop)](https://coveralls.io/github/Bihaqo/t3f?branch=develop)

TensorFlow implementation of the Tensor Train (TT) -Toolbox.

# Installation
T3f assumes you have a working TensorFlow [v1.0](https://www.tensorflow.org/versions/r1.0/install/) or [v1.1](https://www.tensorflow.org/versions/r1.1/install/) installation.
We don't include it into pip requirements since the installation of TensorFlow varies depending on your setup.
Then simply run
```bash
pip install t3f
```

# Basic usage
Import the libraries
```python
import tensorflow as tf
import t3f
```

Generate a random tensor and compute its norm.
```python
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
```

### Arithmetic
```python
a = t3f.random_tensor((3, 2, 2), tt_rank=3)
b_dense = tf.random_normal((3, 2, 2))
# Use TT-SVD on b_dense.
b_tt = t3f.to_tt_tensor(b_dense, max_tt_rank=4)
sum_round = t3f.round(t3f.add(a, b_tt), max_tt_rank=2)
```

### Tensor operations
```python
# Inner product (sum of products of all elements).
a = t3f.random_tensor((3, 2, 2), tt_rank=3)
b = t3f.random_tensor((3, 2, 2), tt_rank=4)
inner_prod = t3f.tt_tt_flat_inner(a, b)
```

### Matrix operations
```python
A = t3f.random_matrix(((3, 2, 2), (2, 3, 3)), tt_rank=3)
b = t3f.random_matrix(((2, 3, 3), None), tt_rank=3)
# Matrix-by-vector
matvec = t3f.matmul(A, b)

# Matrix-by-dense matrix
b_dense = tf.random_normal((18, 1))
matvec2 = t3f.matmul(A, b_dense)
```


# TensorNet example
As an example lets create a neural network with a TT-matrix as a fully-connected layer:
```python
def build_tt_model(x):
  # A 784 x 625 TT-matrix.
  matrix_shape = ((4, 7, 4, 7), (5, 5, 5, 5))
  tt_W_1 = t3f.get_variable('tt_W_1', initializer=t3f.random_matrix(matrix_shape))
  h_1 = tf.nn.relu(t3f.matmul(tt_W_1, x))
  W_2 = tf.get_variable('W_2', shape=[625, 10])
  y = tf.matmul(W_2, h_1)
  return y
```

If you want to start from already trained network and compress its fully-connected layer, you may load the network, find the closes approximation of the existing layer matrix in the TT-format, and then finetune the model
```python
y = build_tt_model(x)
with tf.variable_scope("", reuse=True):
  tt_W_1 = t3f.get_tt_variable('tt_W_1')
W_1 = tf.get_variable('W_1', shape=[784, 625])
tt_init_op = t3f.initialize_from_tensor(tt_W_1, W_1)
loss = tf.nn.softmax_cross_entropy_with_logits(y, labels)
train_step = tf.train.Adam(0.01).minimize(loss)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  restore_all_saved(sess, 'checkpoint/path')
  sess.run(tt_init_op)
  # Finally do the finetuning.
  ...
```
where
```python
def restore_all_saved(sess, path):
  reader = tf.train.NewCheckpointReader(path)
  var_names_in_checkpoint = reader.get_variable_to_shape_map().keys()
  with tf.variable_scope('', reuse=True):
    vars_in_checkpoint = [tf.get_variable(name) for name in var_names_in_checkpoint]
  restorer = tf.train.Saver(var_list=vars_in_checkpoint)
  restorer.restore(sess, path)
```

# Tests
```bash
nosetests  --logging-level=WARNING
```

# Other implementations
There are also implementations of the TT-toolbox in [plain Python](https://github.com/oseledets/ttpy) and [Matlab](https://github.com/oseledets/TT-Toolbox).