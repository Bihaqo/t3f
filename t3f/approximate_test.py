import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from t3f import ops
from t3f import approximate
from t3f import initializers


class _ApproximateTest():

  def testAddN(self):
    # Sum a bunch of TT-matrices.
    tt_a = initializers.random_matrix(((2, 1, 4), (2, 2, 2)), tt_rank=2,
                                      dtype=self.dtype)
    tt_b = initializers.random_matrix(((2, 1, 4), (2, 2, 2)),
                                      tt_rank=[1, 2, 4, 1], dtype=self.dtype)

    def desired(tt_objects):
      res = tt_objects[0]
      for tt in tt_objects[1:]:
        res += tt
      return res

    res_actual = ops.full(approximate.add_n([tt_a, tt_b], 6))
    res_desired = ops.full(desired([tt_a, tt_b]))
    res_desired_val, res_actual_val = self.evaluate([res_desired, res_actual])
    self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

    res_actual = ops.full(approximate.add_n([tt_a, tt_b, tt_a], 8))
    res_desired = ops.full(desired([tt_a, tt_b, tt_a]))
    res_desired_val, res_actual_val = self.evaluate([res_desired, res_actual])
    self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

    res_actual = ops.full(approximate.add_n([tt_a, tt_b, tt_a, tt_a, tt_a], 12))
    res_desired = ops.full(desired([tt_a, tt_b, tt_a, tt_a, tt_a]))
    res_desired_val, res_actual_val = self.evaluate([res_desired, res_actual])
    self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

  def testReduceSumBatch(self):
    # Sum a batch of TT-tensors.

    def desired(tt_batch):
      res = tt_batch[0]
      for i in range(1, tt_batch.batch_size):
        res += tt_batch[i]
      return res
    for batch_size in [2, 3, 4, 5]:
      tt_batch = initializers.random_tensor_batch((4, 3, 5),
                                                  tt_rank=2,
                                                  batch_size=batch_size,
                                                  dtype=self.dtype)
      res_actual = ops.full(approximate.reduce_sum_batch(tt_batch, 10))
      res_desired = ops.full(desired(tt_batch))
      res_desired_val, res_actual_val = self.evaluate([res_desired, res_actual])
      self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

  def testReduceSumBatchWeighted(self):
    # Weighted sum of a batch of TT-tensors.

    def desired(tt_batch, coef):
      res = coef[0] * tt_batch[0]
      for i in range(1, tt_batch.batch_size):
        res += coef[i] * tt_batch[i]
      return res
    tt_batch = initializers.random_tensor_batch((4, 3, 5),
                                                tt_rank=3,
                                                batch_size=3,
                                                dtype=self.dtype)
    res_actual = ops.full(approximate.reduce_sum_batch(tt_batch, 9,
                                                       [1.2, -0.2, 1]))
    res_desired = ops.full(desired(tt_batch, [1.2, -0.2, 1]))
    res_desired_val, res_actual_val = self.evaluate([res_desired, res_actual])
    self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

  def testReduceSumBatchMultipleWeighted(self):
    # Multiple weighted sums of a batch of TT-tensors.

    def desired(tt_batch, coef):
      res = coef[0] * tt_batch[0]
      for i in range(1, tt_batch.batch_size):
        res += coef[i] * tt_batch[i]
      return res
    tt_batch = initializers.random_tensor_batch((4, 3, 5), tt_rank=2,
                                                batch_size=3,
                                                dtype=self.dtype)
    coef = [[1., 0.1],
            [0.9, -0.2],
            [0.3, 0.3]]
    coef = np.array(coef)
    res_actual = ops.full(approximate.reduce_sum_batch(tt_batch, 6,
                                                       coef))
    res_desired_1 = ops.full(desired(tt_batch, coef[:, 0]))
    res_desired_2 = ops.full(desired(tt_batch, coef[:, 1]))
    res_desired = tf.stack((res_desired_1, res_desired_2))
    res_desired_val, res_actual_val = self.evaluate([res_desired, res_actual])
    self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)


class ApproximateTestFloat32(tf.test.TestCase, _ApproximateTest):
  dtype = tf.float32


class ApproximateTestFloat64(tf.test.TestCase, _ApproximateTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
