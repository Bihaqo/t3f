import tensorflow as tf

from t3f import tensor_train
from t3f import initializers
from t3f import ops


class TensorTrainTest(tf.test.TestCase):

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
      a = tf.random_normal((tt_ranks[0], 10, tt_ranks[1]))
      b = tf.random_normal((tt_ranks[2], 9, tt_ranks[3]))
      with self.test_session():
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
        b_new = tf.cast(b, tf.float64)
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
      a = tf.random_normal((tt_ranks[0], 10, tt_ranks[1]))
      b = tf.random_normal((tt_ranks[2], 1, tt_ranks[3]))
      c = tf.random_normal((tt_ranks[4], 2, tt_ranks[5]))
      with self.test_session():
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
        b_new = tf.cast(b, tf.float64)
        actual = tensor_train._are_tt_cores_valid((a, b_new, c), (10, 1, 2),
                                                  claimed_tt_ranks)
        self.assertEqual(False, actual)
        with self.assertRaises(ValueError):
          tensor_train.TensorTrain((a, b_new, c), (10, 1, 2), claimed_tt_ranks)

  def testTensorIndexing(self):
    tens = initializers.random_tensor((3, 3, 4))
    with self.test_session() as sess:
      desired = ops.full(tens)[:, :, :]
      actual = ops.full(tens[:, :, :])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[1, :, :]
      actual = ops.full(tens[1, :, :])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[1:2, 1, :]
      actual = ops.full(tens[1:2, 1, :])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[0:3, :, 3]
      actual = ops.full(tens[0:3, :, 3])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)
      desired = ops.full(tens)[1, :, 3]
      actual = ops.full(tens[1, :, 3])
      desired, actual = sess.run([desired, actual])
      self.assertAllClose(desired, actual)

      # Wrong number of dims.
      with self.assertRaises(ValueError):
        tens[1, :, 3, :]
      with self.assertRaises(ValueError):
        tens[1, 1]

  def testPlaceholderTensorIndexing(self):
    tens = initializers.random_tensor((3, 3, 4))
    with self.test_session() as sess:
      start = tf.placeholder(tf.int32)
      end = tf.placeholder(tf.int32)
      desired = ops.full(tens)[1:3, 1, :3]
      actual = ops.full(tens[start:end, start, :end])
      desired, actual = sess.run([desired, actual], {start: 1, end: 3})
      self.assertAllClose(desired, actual)

if __name__ == "__main__":
  tf.test.main()
