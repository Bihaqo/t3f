import numpy as np
import tensorflow as tf

from t3f import initializers
from t3f import ops


class InitializersTest(tf.test.TestCase):

  def testTensorOnesAndZeros(self):
    tt_ones = initializers.tensor_ones([2, 3, 4])
    tt_zeros = initializers.tensor_zeros([2, 3, 4])

    ones_desired = np.ones((2, 3, 4))
    zeros_desired = np.zeros((2, 3, 4))
    with self.test_session() as sess:
      tt_ones_full = sess.run(ops.full(tt_ones))
      tt_zeros_full = sess.run(ops.full(tt_zeros))
      self.assertAllClose(tt_ones_full, ones_desired)
      self.assertAllClose(tt_zeros_full, zeros_desired)
    bad_shapes = [[[2, 3]], [-1.0, 3]]
    for shape in bad_shapes:
      with self.assertRaises(ValueError):
        initializers.tensor_ones(shape)
        initializers.tensor_zeros(shape)

  def testMatrixOnesAndZeros(self):
    tt_ones = initializers.matrix_ones([[2, 3, 4], [1, 2, 5]])
    tt_zeros = initializers.matrix_zeros([[2, 3, 4], [1, 2, 5]])

    ones_desired = np.ones((24, 10))
    zeros_desired = np.zeros((24, 10))

    bad_shapes = [[[-1, 2, 3], [3, 4, 6]], [[1.5, 2, 4], [2, 5, 6]],
                  [[1], [2, 3]]]
    with self.test_session() as sess:
      tt_ones_full = sess.run(ops.full(tt_ones))
      tt_zeros_full = sess.run(ops.full(tt_zeros))
      self.assertAllClose(tt_ones_full, ones_desired)
      self.assertAllClose(tt_zeros_full, zeros_desired)
    for shape in bad_shapes:
      with self.assertRaises(ValueError):
        initializers.matrix_ones(shape)
        initializers.matrix_zeros(shape)

  def testEye(self):
      tt_eye = initializers.eye([4, 5, 6])
      eye_desired = np.eye(120)
      with self.test_session() as sess:
        eye_full = sess.run(ops.full(tt_eye))
        self.assertAllClose(eye_full, eye_desired)

  def testOnesLikeAndZerosLike(self):
    a = initializers.random_tensor([2, 3, 4])
    b = initializers.ones_like(a)
    c = initializers.zeros_like(a)
    var_list = [ops.full(b), ops.full(c)]
    with self.test_session() as sess:
      bf, cf = sess.run(var_list)
      self.assertAllClose(bf, np.ones((2, 3, 4)))
      self.assertAllClose(cf, np.zeros((2, 3, 4)))
    with self.assertRaises(ValueError):
      initializers.ones_like(1)
    with self.assertRaises(ValueError):
      initializers.zeros_like(1)

  def testRandomTensor(self):
    shapes = [[1, -2], [1.1, 2], [3, 4]]
    tt_ranks = [-1, 1.5, [-2, 3]]
    for shape in shapes:
      for ranks in tt_ranks:
        with self.assertRaises(ValueError):
          initializers.random_tensor(shape, tt_rank=ranks)

  def testRandomMatrix(self):
    shapes = [[[1, -2], None], [[1.1, 2], [3, 4]], [[1], [2, 3]],
              [[1, 2], [3, 4]]]
    tt_ranks = [-1, 1.5, [-2, 3]]
    for shape in shapes:
      for ranks in tt_ranks:
        with self.assertRaises(ValueError):
          initializers.random_matrix(shape, tt_rank=ranks)

  def testRandomTensorBatch(self):
    shapes = [[1, -2], [1.1, 2], [3, 4]]
    tt_ranks = [-1, 1.5, [-2, 3], [7, 8]]
    bs = [-1, 0.5]
    for shape in shapes:
      for ranks in tt_ranks:
        for b in bs:
          with self.assertRaises(ValueError):
            initializers.random_tensor_batch(shape, tt_rank=ranks,
                                             batch_size=b)

  def testRandomMatrixBatch(self):
    shapes = [[[1, -2], None], [[1.1, 2], [3, 4]], [[1], [2, 3]],
              [[1, 2], [3, 4]]]
    tt_ranks = [-1, 1.5, [-2, 3], 5]
    bs = [-1, 0.5]
    for shape in shapes:
      for ranks in tt_ranks:
        for b in bs:
          with self.assertRaises(ValueError):
            initializers.random_matrix_batch(shape, tt_rank=ranks,
                                             batch_size=b)


if __name__ == "__main__":
  tf.test.main()
