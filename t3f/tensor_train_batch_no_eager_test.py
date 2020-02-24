import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from t3f import initializers
from t3f import ops


class _TensorTrainBatchTest():

  def testPlaceholderTensorIndexing(self):
    tens = initializers.random_tensor_batch((3, 3, 4), batch_size=3,
                                            dtype=self.dtype)
    with tf.Session() as sess:
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


class TensorTrainBatchTestFloat32(tf.test.TestCase, _TensorTrainBatchTest):
  dtype = tf.float32


class TensorTrainBatchTestFloat64(tf.test.TestCase, _TensorTrainBatchTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()