import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from t3f import ops
from t3f import batch_ops
from t3f import initializers


class _BatchOpsTest():

  def testConcatMatrix(self):
    # Test concating TTMatrix batches along batch dimension.
    first = initializers.random_matrix_batch(((2, 3), (3, 3)), batch_size=1,
                                             dtype=self.dtype)
    second = initializers.random_matrix_batch(((2, 3), (3, 3)), batch_size=4,
                                              dtype=self.dtype)
    third = initializers.random_matrix_batch(((2, 3), (3, 3)), batch_size=3,
                                             dtype=self.dtype)
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
    res = self.evaluate((first_res, first_second_res, first_second_third_res,
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

  def testBatchMultiply(self):
    # Test multiplying batch of TTMatrices by individual numbers.
    tt = initializers.random_matrix_batch(((2, 3), (3, 3)), batch_size=3,
                                          dtype=self.dtype)
    weights = [0.1, 0, -10]
    actual = batch_ops.multiply_along_batch_dim(tt, weights)
    individual_desired = [weights[i] * tt[i:i+1] for i in range(3)]
    desired = batch_ops.concat_along_batch_dim(individual_desired)
    desired_val, acutual_val = self.evaluate((ops.full(desired), ops.full(actual)))
    self.assertAllClose(desired_val, acutual_val)

  def testGramMatrix(self):
    # Test Gram Matrix of a batch of TT vectors.
    tt_vectors = initializers.random_matrix_batch(((2, 3), None), batch_size=5,
                                                  dtype=self.dtype)
    res_actual = batch_ops.gram_matrix(tt_vectors)
    full_vectors = tf.reshape(ops.full(tt_vectors), (5, 6))
    res_desired = tf.matmul(full_vectors, tf.transpose(full_vectors))
    res_desired = tf.squeeze(res_desired)
    res_actual_val, res_desired_val = self.evaluate((res_actual, res_desired))
    self.assertAllClose(res_desired_val, res_actual_val)

  def testGramMatrixWithMatrix(self):
    # Test Gram Matrix of a batch of TT vectors with providing a matrix, so we
    # should compute
    # res[i, j] = tt_vectors[i] ^ T * matrix * tt_vectors[j]
    tt_vectors = initializers.random_matrix_batch(((2, 3), None), batch_size=4,
                                                  dtype=self.dtype)
    matrix = initializers.random_matrix(((2, 3), (2, 3)), dtype=self.dtype)
    res_actual = batch_ops.gram_matrix(tt_vectors, matrix)
    full_vectors = tf.reshape(ops.full(tt_vectors), (4, 6))
    res = self.evaluate((res_actual, full_vectors, ops.full(matrix)))
    res_actual_val, vectors_val, matrix_val = res
    res_desired_val = np.zeros((4, 4))
    for i in range(4):
      for j in range(4):
        curr_val = np.dot(vectors_val[i], matrix_val)
        curr_val = np.dot(curr_val, vectors_val[j])
        res_desired_val[i, j] = curr_val
    self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

  def testPairwiseFlatInnerTensor(self):
    # Test pairwise_flat_inner of a batch of TT tensors.
    tt_tensors_1 = initializers.random_tensor_batch((2, 3, 2), batch_size=5,
                                                    dtype=self.dtype)
    tt_tensors_2 = initializers.random_tensor_batch((2, 3, 2), batch_size=5,
                                                    dtype=self.dtype)
    res_actual = batch_ops.pairwise_flat_inner(tt_tensors_1, tt_tensors_2)
    full_tensors_1 = tf.reshape(ops.full(tt_tensors_1), (5, 12))
    full_tensors_2 = tf.reshape(ops.full(tt_tensors_2), (5, 12))
    res_desired = tf.matmul(full_tensors_1, tf.transpose(full_tensors_2))
    res_desired = tf.squeeze(res_desired)
    res_actual_val, res_desired_val = self.evaluate((res_actual, res_desired))
    self.assertAllClose(res_desired_val, res_actual_val)

  def testPairwiseFlatInnerMatrix(self):
    # Test pairwise_flat_inner of a batch of TT matrices.
    tt_vectors_1 = initializers.random_matrix_batch(((2, 3), (2, 3)),
                                                    batch_size=5,
                                                    dtype=self.dtype)
    tt_vectors_2 = initializers.random_matrix_batch(((2, 3), (2, 3)),
                                                    batch_size=5,
                                                    dtype=self.dtype)
    res_actual = batch_ops.pairwise_flat_inner(tt_vectors_1, tt_vectors_2)
    full_vectors_1 = tf.reshape(ops.full(tt_vectors_1), (5, 36))
    full_vectors_2 = tf.reshape(ops.full(tt_vectors_2), (5, 36))
    res_desired = tf.matmul(full_vectors_1, tf.transpose(full_vectors_2))
    res_desired = tf.squeeze(res_desired)
    res_actual_val, res_desired_val = self.evaluate((res_actual, res_desired))
    self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

  def testPairwiseFlatInnerVectorsWithMatrix(self):
    # Test pairwise_flat_inner of a batch of TT vectors with providing a matrix,
    # so we should compute
    # res[i, j] = tt_vectors[i] ^ T * matrix * tt_vectors[j]
    tt_vectors_1 = initializers.random_matrix_batch(((2, 3), None),
                                                    batch_size=2,
                                                    dtype=self.dtype)
    tt_vectors_2 = initializers.random_matrix_batch(((2, 3), None),
                                                    batch_size=3,
                                                    dtype=self.dtype)
    matrix = initializers.random_matrix(((2, 3), (2, 3)), dtype=self.dtype)
    res_actual = batch_ops.pairwise_flat_inner(tt_vectors_1, tt_vectors_2,
                                               matrix)
    full_vectors_1 = tf.reshape(ops.full(tt_vectors_1), (2, 6))
    full_vectors_2 = tf.reshape(ops.full(tt_vectors_2), (3, 6))
    res = self.evaluate((res_actual, full_vectors_1, full_vectors_2,
                    ops.full(matrix)))
    res_actual_val, vectors_1_val, vectors_2_val, matrix_val = res
    res_desired_val = np.zeros((2, 3))
    for i in range(2):
      for j in range(3):
        curr_val = np.dot(vectors_1_val[i], matrix_val)
        curr_val = np.dot(curr_val, vectors_2_val[j])
        res_desired_val[i, j] = curr_val
    self.assertAllClose(res_desired_val, res_actual_val)


class BatchOpsTestFloat32(tf.test.TestCase, _BatchOpsTest):
  dtype = tf.float32


class BatchOpsTestFloat64(tf.test.TestCase, _BatchOpsTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()

