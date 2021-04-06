import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import ops
from t3f import initializers
from t3f import variables
from t3f import kronecker as kr

class _KroneckerTest():

  def testIsKronNonKron(self):
    # Tests _is_kron on a non-Kronecker matrix
    initializer = initializers.random_matrix(((2, 3), (3, 2)), tt_rank=2,
                                             dtype=self.dtype)
    tt_mat = variables.get_variable('tt_mat', initializer=initializer)
    self.assertFalse(kr._is_kron(tt_mat))

  def testIsKronKron(self):
    # Tests _is_kron on a Kronecker matrix
    initializer = initializers.random_matrix(((2, 3), (3, 2)), tt_rank=1,
                                             dtype=self.dtype)
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    self.assertTrue(kr._is_kron(kron_mat))

  def testDet(self):
    # Tests the determinant function
    initializer = initializers.random_matrix(((2, 3, 2), (2, 3, 2)), tt_rank=1,
                                             dtype=self.dtype)
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    init_op = tf.compat.v1.global_variables_initializer()
    self.evaluate(init_op)
    desired = np.linalg.det(self.evaluate(ops.full(kron_mat)))
    actual = self.evaluate(kr.determinant(kron_mat))
    self.assertAllClose(desired, actual)

  def testSlogDet(self):
    # Tests the slog_determinant function
    
    # TODO: use kron and -1 * kron matrices, when mul is implemented 
    # the current version is platform-dependent
    
    tf.compat.v1.set_random_seed(5) # negative derminant
    initializer = initializers.random_matrix(((2, 3), (2, 3)), tt_rank=1,
                                             dtype=self.dtype)
    kron_neg = variables.get_variable('kron_neg', initializer=initializer)
  
    tf.compat.v1.set_random_seed(1) # positive determinant
    initializer = initializers.random_matrix(((2, 3), (2, 3)), tt_rank=1,
                                             dtype=self.dtype)
    kron_pos = variables.get_variable('kron_pos', initializer=initializer)

    init_op = tf.compat.v1.global_variables_initializer()
     # negative derminant
    self.evaluate(init_op)
    desired_sign, desired_det = np.linalg.slogdet(self.evaluate(ops.full(kron_neg)))
    actual_sign, actual_det = self.evaluate(kr.slog_determinant(kron_neg))
    self.assertEqual(desired_sign, actual_sign)
    self.assertAllClose(desired_det, actual_det)

    # positive determinant 
    desired_sign, desired_det = np.linalg.slogdet(self.evaluate(ops.full(kron_pos)))
    actual_sign, actual_det = self.evaluate(kr.slog_determinant(kron_pos))
    self.assertEqual(desired_sign, actual_sign)
    self.assertAllClose(desired_det, actual_det)

  def testInv(self):
    # Tests the inv function
    initializer = initializers.random_matrix(((2, 3, 2), (2, 3, 2)), tt_rank=1,
                                             dtype=self.dtype)
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    init_op = tf.compat.v1.global_variables_initializer()
    self.evaluate(init_op)
    desired = np.linalg.inv(self.evaluate(ops.full(kron_mat)))
    actual = self.evaluate(ops.full(kr.inv(kron_mat)))
    self.assertAllClose(desired, actual)

  def testCholesky(self):
    # Tests the cholesky function
    np.random.seed(8)

    # generating two symmetric positive-definite tt-cores
    L_1 = np.tril(np.random.normal(scale=2., size=(2, 2)))
    L_1 = L_1.astype(self.dtype.as_numpy_dtype)
    L_2 = np.tril(np.random.normal(scale=2., size=(3, 3)))
    L_2 = L_2.astype(self.dtype.as_numpy_dtype)
    K_1 = L_1.dot(L_1.T)
    K_2 = L_2.dot(L_2.T)
    K = np.kron(K_1, K_2)
    initializer = TensorTrain([K_1[None, :, :, None], 
                               K_2[None, :, :, None]], 
                              tt_ranks=7*[1])
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    init_op = tf.compat.v1.global_variables_initializer()
    self.evaluate(init_op)
    desired = np.linalg.cholesky(K)
    actual = self.evaluate(ops.full(kr.cholesky(kron_mat)))
    self.assertAllClose(desired, actual, atol=1e-5, rtol=1e-5)

class _BatchKroneckerTest():

  def testIsKronNonKron(self):
    # Tests _is_kron on a non-Kronecker matrix batch
    initializer = initializers.random_matrix_batch(((2, 3), (3, 2)), tt_rank=2, 
                                                   batch_size=3,
                                                   dtype=self.dtype)
    tt_mat_batch = variables.get_variable('tt_mat_batch', 
                                          initializer=initializer)
    self.assertFalse(kr._is_kron(tt_mat_batch))
 
  def testIsKronKron(self):
    # Tests _is_kron on a Kronecker matrix batch
    initializer = initializers.random_matrix_batch(((2, 3), (3, 2)), tt_rank=1, 
                                                   batch_size=3,
                                                   dtype=self.dtype)
    kron_mat_batch = variables.get_variable('kron_mat_batch', 
                                            initializer=initializer)
    self.assertTrue(kr._is_kron(kron_mat_batch))

  def testDet(self):
    # Tests the determinant function
    initializer = initializers.random_matrix_batch(((2, 3, 2), (2, 3, 2)), 
                                                   tt_rank=1, batch_size=3,
                                                   dtype=self.dtype)
    kron_mat_batch = variables.get_variable('kron_mat_batch', 
                                            initializer=initializer)
    init_op = tf.compat.v1.global_variables_initializer()
    self.evaluate(init_op)
    desired = self.evaluate(tf.linalg.det(ops.full(kron_mat_batch)))
    actual = self.evaluate(kr.determinant(kron_mat_batch))
    self.assertAllClose(desired, actual)

  def testSlogDet(self):
    # Tests the slog_determinant function
    
    tf.compat.v1.set_random_seed(1) # negative and positive determinants
    initializer = initializers.random_matrix_batch(((2, 3), (2, 3)), tt_rank=1, 
                                                   batch_size=3,
                                                   dtype=self.dtype)
    kron_mat_batch = variables.get_variable('kron_mat_batch', 
                                            initializer=initializer)
  
    init_op = tf.compat.v1.global_variables_initializer()
     # negative derminant
    self.evaluate(init_op)
    desired_sign, desired_det = np.linalg.slogdet(
        self.evaluate(ops.full(kron_mat_batch)))
    actual_sign, actual_det = self.evaluate(kr.slog_determinant(kron_mat_batch))
    self.assertAllEqual(desired_sign, actual_sign)
    self.assertAllClose(desired_det, actual_det)

  def testInv(self):
    # Tests the inv function
    initializer = initializers.random_matrix_batch(((2, 3, 2), (2, 3, 2)), 
                                                   tt_rank=1, batch_size=3,
                                                   dtype=self.dtype)
    kron_mat_batch = variables.get_variable('kron_mat_batch', 
                                            initializer=initializer)
    init_op = tf.compat.v1.global_variables_initializer()
    self.evaluate(init_op)
    desired = np.linalg.inv(self.evaluate(ops.full(kron_mat_batch)))
    actual = self.evaluate(ops.full(kr.inv(kron_mat_batch)))
    self.assertAllClose(desired, actual, atol=1e-4)

  def testCholesky(self):
    # Tests the cholesky function
    np.random.seed(8)

    # generating two symmetric positive-definite tt-cores
    L_1 = np.tril(np.random.normal(scale=2., size=(4, 2, 2)))
    L_1 = L_1.astype(self.dtype.as_numpy_dtype)
    L_2 = np.tril(np.random.normal(scale=2., size=(4, 3, 3)))
    L_2 = L_2.astype(self.dtype.as_numpy_dtype)
    K_1 = np.einsum('ijk,ilk->ijl', L_1, L_1)
    K_2 = np.einsum('ijk,ilk->ijl', L_2, L_2)
    initializer = TensorTrainBatch([K_1[:, None, :, :, None],
                                    K_2[:, None, :, :, None]], tt_ranks=7*[1])
    kron_mat_batch = variables.get_variable('kron_mat_batch', 
                                            initializer=initializer)
    init_op = tf.compat.v1.global_variables_initializer()
    self.evaluate(init_op)
    desired = np.linalg.cholesky(self.evaluate(ops.full(kron_mat_batch)))
    actual = self.evaluate(ops.full(kr.cholesky(kron_mat_batch)))
    self.assertAllClose(desired, actual)


class KroneckerTestFloat32(tf.test.TestCase, _KroneckerTest):
  dtype = tf.float32


class KroneckerTestFloat64(tf.test.TestCase, _KroneckerTest):
  dtype = tf.float64


class BatchKroneckerTestFloat32(tf.test.TestCase, _BatchKroneckerTest):
  dtype = tf.float32


class BatchKroneckerTestFloat64(tf.test.TestCase, _BatchKroneckerTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
