# Graph mode tests.
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from t3f import ops
from t3f import batch_ops
from t3f import initializers


class _BatchOpsTest():

  def testConcatTensorPlaceholders(self):
    # Test concating TTTensors of unknown batch sizes along batch dimension.
    number_of_objects = tf.placeholder(tf.int32)
    all = initializers.random_tensor_batch((2, 3), batch_size=5,
                                           dtype=self.dtype)
    actual = batch_ops.concat_along_batch_dim((all[:number_of_objects],
                                              all[number_of_objects:]))
    with tf.Session() as sess:
      desired_val, actual_val = sess.run((ops.full(all), ops.full(actual)),
                                         feed_dict={number_of_objects: 2})
      self.assertAllClose(desired_val, actual_val)

  def testConcatMatrixPlaceholders(self):
    # Test concating TTMatrices of unknown batch sizes along batch dimension.
    number_of_objects = tf.placeholder(tf.int32)
    all = initializers.random_matrix_batch(((2, 3), (2, 3)), batch_size=5,
                                           dtype=self.dtype)
    actual = batch_ops.concat_along_batch_dim((all[:number_of_objects],
                                              all[number_of_objects:]))
    with tf.Session() as sess:
      desired_val, actual_val = sess.run((ops.full(all), ops.full(actual)),
                                         feed_dict={number_of_objects: 2})
      self.assertAllClose(desired_val, actual_val)


class BatchOpsTestFloat32(tf.test.TestCase, _BatchOpsTest):
  dtype = tf.float32


class BatchOpsTestFloat64(tf.test.TestCase, _BatchOpsTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
