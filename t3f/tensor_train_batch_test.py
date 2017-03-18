import tensorflow as tf

from tensor_train import TensorTrain
import tensor_train_batch
import initializers
import ops


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

if __name__ == "__main__":
  tf.test.main()
