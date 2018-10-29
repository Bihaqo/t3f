import numpy as np
import tensorflow as tf

from t3f import initializers
from t3f import ops


class _InitializersTest():

  def testTensorOnesAndZeros(self):
    tt_ones = initializers.tensor_ones([2, 3, 4], dtype=self.tf_dtype)
    tt_zeros = initializers.tensor_zeros([2, 3, 4], dtype=self.tf_dtype)

    ones_desired = np.ones((2, 3, 4), dtype=self.tf_dtype.as_numpy_dtype)
    zeros_desired = np.zeros((2, 3, 4), dtype=self.tf_dtype.as_numpy_dtype)
    with self.test_session() as sess:
      tt_ones_full = sess.run(ops.full(tt_ones))
      tt_zeros_full = sess.run(ops.full(tt_zeros))
      self.assertAllClose(tt_ones_full, ones_desired)
      self.assertEqual(tt_ones_full.dtype, ones_desired.dtype)
      self.assertAllClose(tt_zeros_full, zeros_desired)
      self.assertEqual(tt_zeros_full.dtype, zeros_desired.dtype)
    bad_shapes = [[[2, 3]], [-1, 3], [0.1, 4]]
    for shape in bad_shapes:
      with self.assertRaises(ValueError):
        initializers.tensor_ones(shape)
      with self.assertRaises(ValueError):
        initializers.tensor_zeros(shape)

  def testMatrixOnesAndZeros(self):
    tt_ones = initializers.matrix_ones([[2, 3, 4], [1, 2, 5]],
                                       dtype=self.tf_dtype)
    tt_zeros = initializers.matrix_zeros([[2, 3, 4], [1, 2, 5]],
                                         dtype=self.tf_dtype)

    ones_desired = np.ones((24, 10), dtype=self.tf_dtype.as_numpy_dtype)
    zeros_desired = np.zeros((24, 10), dtype=self.tf_dtype.as_numpy_dtype)

    bad_shapes = [[[-1, 2, 3], [3, 4, 6]], [[1.5, 2, 4], [2, 5, 6]],
                  [[1], [2, 3]], [2, 3, 4]]
    with self.test_session() as sess:
      tt_ones_full = sess.run(ops.full(tt_ones))
      tt_zeros_full = sess.run(ops.full(tt_zeros))
      self.assertAllClose(tt_ones_full, ones_desired)
      self.assertEqual(tt_ones_full.dtype, ones_desired.dtype)
      self.assertAllClose(tt_zeros_full, zeros_desired)
      self.assertEqual(tt_zeros_full.dtype, zeros_desired.dtype)
    for shape in bad_shapes:
      with self.assertRaises(ValueError):
        initializers.matrix_ones(shape)
      with self.assertRaises(ValueError):
        initializers.matrix_zeros(shape)

  def testEye(self):
      tt_eye = initializers.eye([4, 5, 6], dtype=self.tf_dtype)
      eye_desired = np.eye(120)
      with self.test_session() as sess:
        eye_full = sess.run(ops.full(tt_eye))
        self.assertAllClose(eye_full, eye_desired)
      bad_shapes = [[[2, 3]], [-1, 3], [0.1, 4]]
      for shape in bad_shapes:
        with self.assertRaises(ValueError):
          initializers.eye(shape)

  def testOnesLikeAndZerosLike(self):
    a = initializers.random_tensor([2, 3, 4], dtype=self.tf_dtype)
    b = initializers.ones_like(a)
    c = initializers.zeros_like(a)
    var_list = [ops.full(b), ops.full(c)]
    with self.test_session() as sess:
      bf, cf = sess.run(var_list)
      self.assertAllClose(bf, np.ones((2, 3, 4)))
      self.assertEqual(self.tf_dtype.as_numpy_dtype, bf.dtype)
      self.assertAllClose(cf, np.zeros((2, 3, 4)))
      self.assertEqual(self.tf_dtype.as_numpy_dtype, cf.dtype)
    with self.assertRaises(ValueError):
      initializers.ones_like(1)
    with self.assertRaises(ValueError):
      initializers.zeros_like(1)

  def testInvalidShapeOrRanksTensor(self):
    shapes = [[3, 4], [3, 4], [3, 4], [3, 4], [1, -2], [1.1, 2], [[3, 4]]]
    tt_ranks = [-2, 1.5, [2, 3, 4, 5], [1.5], 2, 2, 2]
    bad_cases = zip(shapes, tt_ranks)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.random_tensor(case[0], tt_rank=case[1],
                                   dtype=self.tf_dtype)

    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.tensor_with_random_cores(case[0], tt_rank=case[1],
                                              dtype=self.tf_dtype)

  def testInvalidShapeOrRanksMatrix(self):
    shapes = [[1, 2, 3], [[1, 2], [1, 2, 3]], [[-1, 2, 3], [1, 2, 3]],
              [[0.5, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]]]
    tt_ranks = [2, 2, 2, 2, -1, [[[1]]], [2.5, 3]]
    bad_cases = zip(shapes, tt_ranks)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.random_matrix(case[0], tt_rank=case[1],
                                   dtype=self.tf_dtype)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.matrix_with_random_cores(case[0], tt_rank=case[1],
                                              dtype=self.tf_dtype)

  def testInvalidShapeOrRanksTensorBatch(self):
    shapes = [[3, 4], [3, 4], [3, 4], [3, 4], [1, -2], [1.1, 2], [[3, 4]],
              [1, 2], [3, 4]]
    tt_ranks = [-2, 1.5, [2, 3, 4, 5], [1.5], 2, 2, 2, 2, 2]
    bs = [1] * 7 + [-1] + [0.5]
    bad_cases = zip(shapes, tt_ranks, bs)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.random_tensor_batch(case[0], tt_rank=case[1],
                                         batch_size=case[2],
                                         dtype=self.tf_dtype)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.tensor_batch_with_random_cores(case[0], tt_rank=case[1],
                                                    batch_size=case[2],
                                                    dtype=self.tf_dtype)

  def testInvalidShapeOrRanksMatrixBatch(self):
    shapes = [[1, 2, 3], [[1, 2], [1, 2, 3]], [[-1, 2, 3], [1, 2, 3]],
              [[0.5, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]]]
    tt_ranks = [2, 2, 2, 2, -1, [[[1]]], [2.5, 3], 2, 2]
    bs = 7 * [1] + [-1] + [0.5]
    bad_cases = zip(shapes, tt_ranks, bs)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.random_matrix_batch(case[0], tt_rank=case[1],
                                         batch_size=case[2],
                                         dtype=self.tf_dtype)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.matrix_batch_with_random_cores(case[0], tt_rank=case[1],
                                                    batch_size=case[2],
                                                    dtype=self.tf_dtype)

  def testInvalidShapeOrRanksGlorot(self):
    shapes = [[1, 2, 3], [[1, 2], [1, 2, 3]], [[-1, 2, 3], [1, 2, 3]],
              [[0.5, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]]]
    tt_ranks = [2, 2, 2, 2, -1, [[[1]]], [2.5, 3]]
    bad_cases = zip(shapes, tt_ranks)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.glorot_initializer(case[0], tt_rank=case[1])

  def testHe(self):
    shapes = [[1, 2, 3], [[1, 2], [1, 2, 3]], [[-1, 2, 3], [1, 2, 3]],
              [[0.5, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]]]
    tt_ranks = [2, 2, 2, 2, -1, [[[1]]], [2.5, 3]]
    bad_cases = zip(shapes, tt_ranks)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.he_initializer(case[0], tt_rank=case[1],
                                    dtype=self.tf_dtype)

  def testInvalidShapeOrRanksLecun(self):
    shapes = [[1, 2, 3], [[1, 2], [1, 2, 3]], [[-1, 2, 3], [1, 2, 3]],
              [[0.5, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3]]]
    tt_ranks = [2, 2, 2, 2, -1, [[[1]]], [2.5, 3]]
    bad_cases = zip(shapes, tt_ranks)
    for case in bad_cases:
      with self.assertRaises(ValueError):
        initializers.lecun_initializer(case[0], tt_rank=case[1],
                                       dtype=self.tf_dtype)


class InitializersTestFloat32(tf.test.TestCase, _InitializersTest):
  tf_dtype = tf.float32


class InitializersTestFloat64(tf.test.TestCase, _InitializersTest):
  tf_dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
