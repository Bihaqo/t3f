import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain
from tensor_train_batch import TensorTrainBatch
import ops
import batch_ops
import initializers


class BatchOpsTest(tf.test.TestCase):

  def testGramMatrix(self):
    np.random.seed(1)
    tt_vectors = initializers.random_matrix_batch(((2, 3), None), batch_size=5)
    res_actual = batch_ops.gram_matrix(tt_vectors)
    full_vectors = tf.reshape(ops.full(tt_vectors), (5, 6))
    print(full_vectors.get_shape())
    res_desired = tf.matmul(full_vectors, tf.transpose(full_vectors))
    res_desired = tf.squeeze(res_desired)
    # print(res_desired.get_shape())
    with self.test_session() as sess:
      res_actual_val, res_desired_val = sess.run((res_actual, res_desired))
      self.assertAllClose(res_desired_val, res_actual_val)

if __name__ == "__main__":
  tf.test.main()

