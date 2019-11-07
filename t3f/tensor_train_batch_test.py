import numpy as np
import tensorflow.compat.v1 as tf

from t3f import initializers
from t3f import ops


class _TensorTrainBatchTest():

  def testTensorIndexing(self):
    tens = initializers.random_tensor_batch((3, 3, 4), batch_size=3,
                                            dtype=self.dtype)
    with self.test_session() as sess:
      desired = ops.full(tens)[:, :, :, :]
      actual = ops.full(tens[:, :, :, :])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[1:3, :, :, :]
      actual = ops.full(tens[1:3])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[1, :, :, :]
      actual = ops.full(tens[1])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[2, 1, :, :]
      actual = ops.full(tens[2, 1, :, :])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[2, 1:2, 1, :]
      actual = ops.full(tens[2, 1:2, 1, :])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[1:2, 0:3, :, 3]
      actual = ops.full(tens[1:2, 0:3, :, 3])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[:, 1, :, 3]
      actual = ops.full(tens[:, 1, :, 3])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)

      # Wrong number of dims.
      with self.assertRaises(ValueError):
        tens[1, :, 3]
      with self.assertRaises(ValueError):
        tens[1, :, 3, 1:2, 1:3]
      with self.assertRaises(ValueError):
        tens[1, 1]

  def testPlaceholderTensorIndexing(self):
    tens = initializers.random_tensor_batch((3, 3, 4), batch_size=3,
                                            dtype=self.dtype)
    with self.test_session() as sess:
      start = tf.placeholder(tf.int32)
      end = tf.placeholder(tf.int32)

      desired = ops.full(tens)[0:-1]
      actual = ops.full(tens[start:end])
      desired, actual = sess.run([desired, actual], {start: 0, end: -1})
      self.assertAllClose(desired, actual)

      desired = ops.full(tens)[0:1]
      actual = ops.full(tens[start:end])
      desired, actual = sess.run([desired, actual], {start: 0, end: 1})
      self.assertAllClose(desired, actual)

      desired = ops.full(tens)[1]
      actual = ops.full(tens[start])
      desired, actual = sess.run([desired, actual], {start: 1})
      self.assertAllClose(desired, actual)

      desired = ops.full(tens)[1, 1:3, 1, :3]
      actual = ops.full(tens[start, start:end, start, :end])
      desired, actual = sess.run([desired, actual], {start: 1, end: 3})
      self.assertAllClose(desired, actual)

  def testShapeOverflow(self):
    large_shape = [10] * 20
    tensor = initializers.random_matrix_batch([large_shape, large_shape],
                                              batch_size=5, dtype=self.dtype)
    shape = tensor.get_shape()
    self.assertEqual([5, 10 ** 20, 10 ** 20], shape)


class TensorTrainBatchTestFloat32(tf.test.TestCase, _TensorTrainBatchTest):
  dtype = tf.float32


class TensorTrainBatchTestFloat64(tf.test.TestCase, _TensorTrainBatchTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
