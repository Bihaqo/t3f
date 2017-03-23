import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain
from tensor_train_batch import TensorTrainBatch
import ops
import batch_ops
import initializers


class BatchOpsTest(tf.test.TestCase):

  def testConcatMatrix(self):
    # Test concating TTMatrix batches along batch dimension.
    first = initializers.random_matrix_batch(((2, 3), (3, 3)), batch_size=1)
    second = initializers.random_matrix_batch(((2, 3), (3, 3)), batch_size=4)
    third = initializers.random_matrix_batch(((2, 3), (3, 3)), batch_size=3)
    first_res = batch_ops.concat_along_batch_dim((first))
    first_res = ops.full(first_res)
    first_second_res = batch_ops.concat_along_batch_dim((first, second))
    first_second_res = ops.full(first_second_res)
    first_second_third_res = batch_ops.concat_along_batch_dim((first, second,
                                                               third))
    first_second_third_res = ops.full(first_second_third_res)

    first_full = ops.full(first)
    second_full = ops.full(second)
    third_full = ops.full(third)
    first_desired = first_full
    first_second_desired = tf.concat((first_full, second_full), axis=0)
    first_second_third_desired = tf.concat((first_full, second_full, third_full),
                                           axis=0)
    with self.test_session() as sess:
      res = sess.run((first_res, first_second_res, first_second_third_res,
                      first_desired, first_second_desired,
                      first_second_third_desired))
      first_res_val = res[0]
      first_second_res_val = res[1]
      first_second_third_res_val = res[2]
      first_desired_val = res[3]
      first_second_desired_val = res[4]
      first_second_third_desired_val = res[5]
      self.assertAllClose(first_res_val, first_desired_val)
      self.assertAllClose(first_second_res_val, first_second_desired_val)
      self.assertAllClose(first_second_third_res_val, first_second_third_desired_val)

  def testGramMatrix(self):
    # Test Gram Matrix of a batch of TT vectors.
    tt_vectors = initializers.random_matrix_batch(((2, 3), None), batch_size=5)
    res_actual = batch_ops.gram_matrix(tt_vectors)
    full_vectors = tf.reshape(ops.full(tt_vectors), (5, 6))
    res_desired = tf.matmul(full_vectors, tf.transpose(full_vectors))
    res_desired = tf.squeeze(res_desired)
    with self.test_session() as sess:
      res_actual_val, res_desired_val = sess.run((res_actual, res_desired))
      self.assertAllClose(res_desired_val, res_actual_val)

  def testGramMatrixWithMatrix(self):
    # Test Gram Matrix of a batch of TT vectors with providing a matrix, so we
    # should compute
    # res[i, j] = tt_vectors[i] ^ T * matrix * tt_vectors[j]
    tt_vectors = initializers.random_matrix_batch((None, (2, 3)), batch_size=4)
    matrix = initializers.random_matrix(((2, 3), (2, 3)))
    res_actual = batch_ops.gram_matrix(tt_vectors, matrix)
    full_vectors = tf.reshape(ops.full(tt_vectors), (4, 6))
    with self.test_session() as sess:
      res = sess.run((res_actual, full_vectors, ops.full(matrix)))
      res_actual_val, vectors_val, matrix_val = res
      res_desired_val = np.zeros((4, 4))
      for i in range(4):
        for j in range(4):
          curr_val = np.dot(vectors_val[i], matrix_val)
          curr_val = np.dot(curr_val, vectors_val[j])
          res_desired_val[i, j] = curr_val
      self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
  tf.test.main()

