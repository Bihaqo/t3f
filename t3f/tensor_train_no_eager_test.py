import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from t3f import tensor_train
from t3f import initializers
from t3f import ops


class _TensorTrainTest():

  def testPlaceholderTensorIndexing(self):
    tens = initializers.random_tensor((3, 3, 4), dtype=self.dtype)
    with tf.Session() as sess:
      start = tf.placeholder(tf.int32)
      end = tf.placeholder(tf.int32)
      desired = ops.full(tens)[1:3, 1, :3]
      actual = ops.full(tens[start:end, start, :end])
      desired, actual = sess.run([desired, actual], {start: 1, end: 3})
      self.assertAllClose(desired, actual)


class TensorTrainTestFloat32(tf.test.TestCase, _TensorTrainTest):
  dtype = tf.float32


class TensorTrainTestFloat64(tf.test.TestCase, _TensorTrainTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()