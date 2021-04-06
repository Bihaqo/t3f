import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from t3f import tensor_train
from t3f import initializers
from t3f import ops


class _TensorTrainTest():

  def testValidateTTCores2d(self):
    schedule = (((1, 1, 1, 1), (1, 1, 1), True),
                ((1, 1, 1, 1), None, True),
                ((1, 1, 1, 1), (1, 2, 1), False),
                ((1, 1, 1, 1), (2, 1, 1), False),
                ((1, 2, 2, 1), (1, 2, 1), True),
                ((1, 2, 2, 1), (1, 1, 1), False),
                ((1, 2, 2, 1), (1, 1, 2), False),
                ((1, 2, 2, 1), None, True),
                ((2, 1, 1, 1), None, False),
                ((2, 1, 1, 1), (1, 1, 1), False),
                ((1, 1, 1, 2), None, False),
                ((1, 1, 1, 2), (1, 1, 1), False),
                ((1, 1, 2, 1), None, False),
                ((1, 1, 2, 1), (1, 2, 1), False),
                ((1, 2, 1, 1), None, False),
                ((1, 2, 1, 1), (1, 2, 1), False))

    for tt_ranks, claimed_tt_ranks, desired in schedule:
      a = tf.random.normal((tt_ranks[0], 10, tt_ranks[1]), dtype=self.dtype)
      b = tf.random.normal((tt_ranks[2], 9, tt_ranks[3]), dtype=self.dtype)
      actual = tensor_train._are_tt_cores_valid((a, b), (10, 9),
                                                claimed_tt_ranks)
      self.assertEqual(desired, actual)
      # Wrong shape.
      actual = tensor_train._are_tt_cores_valid((a, b), (9, 9),
                                                claimed_tt_ranks)
      self.assertEqual(False, actual)
      if not desired:
        with self.assertRaises(ValueError):
          tensor_train.TensorTrain((a, b), (10, 9), claimed_tt_ranks)

      # Make dtypes inconsistent.
      b_new = tf.cast(b, tf.float16)
      actual = tensor_train._are_tt_cores_valid((a, b_new), (10, 9),
                                                claimed_tt_ranks)
      self.assertEqual(False, actual)
      with self.assertRaises(ValueError):
        tensor_train.TensorTrain((a, b_new), (10, 9), claimed_tt_ranks)

  def testValidateTTCores3d(self):
    schedule = (((1, 1, 1, 1, 1, 1), (1, 1, 1, 1), True),
                ((1, 1, 1, 1, 1, 1), None, True),
                ((1, 1, 1, 1, 1, 1), (1, 1, 1, 2), False),
                ((1, 1, 1, 1, 1, 1), (1, 1, 2, 1), False),
                ((1, 2, 2, 2, 2, 1), (1, 2, 2, 1), True),
                ((1, 2, 2, 2, 2, 1), None, True),
                ((1, 2, 2, 2, 2, 1), (1, 2, 1, 1), False),
                ((2, 1, 1, 1, 1, 1), None, False),
                ((2, 1, 1, 1, 1, 1), (2, 1, 1, 1), False),
                ((1, 1, 1, 1, 1, 2), None, False),
                ((1, 1, 1, 1, 1, 2), (1, 1, 1, 2), False),
                ((1, 1, 2, 1, 1, 1), None, False),
                ((1, 1, 2, 1, 1, 1), (1, 2, 1, 1), False),
                ((1, 2, 1, 1, 1, 1), None, False),
                ((1, 2, 1, 1, 1, 1), (1, 2, 1, 1), False),
                ((1, 2, 2, 1, 1, 1), (1, 2, 1, 1), True),
                ((1, 2, 2, 1, 1, 1), (1, 2, 2, 1), False),
                ((1, 2, 2, 1, 1, 1), None, True),
                ((1, 2, 2, 3, 3, 1), (1, 2, 3, 1), True),
                ((1, 2, 2, 3, 3, 1), None, True))

    for tt_ranks, claimed_tt_ranks, desired in schedule:
      a = tf.random.normal((tt_ranks[0], 10, tt_ranks[1]), dtype=self.dtype)
      b = tf.random.normal((tt_ranks[2], 1, tt_ranks[3]), dtype=self.dtype)
      c = tf.random.normal((tt_ranks[4], 2, tt_ranks[5]), dtype=self.dtype)
      actual = tensor_train._are_tt_cores_valid((a, b, c), (10, 1, 2),
                                                claimed_tt_ranks)
      self.assertEqual(desired, actual)
      # Wrong shape.
      actual = tensor_train._are_tt_cores_valid((a, b, c), (10, 1, 1),
                                                claimed_tt_ranks)
      self.assertEqual(False, actual)
      if not desired:
        with self.assertRaises(ValueError):
          tensor_train.TensorTrain((a, b, c), (10, 1, 2), claimed_tt_ranks)

      # Make dtypes inconsistent.
      b_new = tf.cast(b, tf.float16)
      actual = tensor_train._are_tt_cores_valid((a, b_new, c), (10, 1, 2),
                                                claimed_tt_ranks)
      self.assertEqual(False, actual)
      with self.assertRaises(ValueError):
        tensor_train.TensorTrain((a, b_new, c), (10, 1, 2), claimed_tt_ranks)

  def testTensorIndexing(self):
    tens = initializers.random_tensor((3, 3, 4), dtype=self.dtype)
    desired = ops.full(tens)[:, :, :]
    actual = ops.full(tens[:, :, :])
    desired, actual = self.evaluate([desired, actual])
    self.assertAllClose(desired, actual)
    desired = ops.full(tens)[1, :, :]
    actual = ops.full(tens[1, :, :])
    desired, actual = self.evaluate([desired, actual])
    self.assertAllClose(desired, actual)
    desired = ops.full(tens)[1:2, 1, :]
    actual = ops.full(tens[1:2, 1, :])
    desired, actual = self.evaluate([desired, actual])
    self.assertAllClose(desired, actual)
    desired = ops.full(tens)[0:3, :, 3]
    actual = ops.full(tens[0:3, :, 3])
    desired, actual = self.evaluate([desired, actual])
    self.assertAllClose(desired, actual)
    desired = ops.full(tens)[1, :, 3]
    actual = ops.full(tens[1, :, 3])
    desired, actual = self.evaluate([desired, actual])
    self.assertAllClose(desired, actual)

    # Wrong number of dims.
    with self.assertRaises(ValueError):
      tens[1, :, 3, :]
    with self.assertRaises(ValueError):
      tens[1, 1]

  def testShapeOverflow(self):
    large_shape = [10] * 20
    matrix = initializers.matrix_zeros([large_shape, large_shape],
                                       dtype=self.dtype)
    shape = matrix.get_shape()
    self.assertEqual([10 ** 20, 10 ** 20], shape)


class TensorTrainTestFloat32(tf.test.TestCase, _TensorTrainTest):
  dtype = tf.float32


class TensorTrainTestFloat64(tf.test.TestCase, _TensorTrainTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
