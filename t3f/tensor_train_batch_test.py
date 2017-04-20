import tensorflow as tf

from t3f import initializers
from t3f import ops


class TensorTrainBatchTest(tf.test.TestCase):

  def testTensorIndexing(self):
    tens = initializers.random_tensor_batch((3, 3, 4), batch_size=3)
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
    tens = initializers.random_tensor_batch((3, 3, 4), batch_size=3)
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


if __name__ == "__main__":
  tf.test.main()
