import numpy as np
import tensorflow as tf

from t3f import tensor_train
from t3f import kronecker as kr
from t3f import ops
from t3f import initializers
from t3f import variables

class KroneckerTest(tf.test.TestCase):

  def testIsKronNonKron(self):
    # Tests _is_kron on a non-Kronecker matrix
    initializer = initializers.random_matrix(((2, 3), (3, 2)), tt_rank=2)
    tt_mat = variables.get_variable('tt_mat', initializer=initializer)
    self.assertFalse(kr._is_kron(tt_mat))
           
  def testIsKronKron(self):
    # Tests _is_kron on a Kronecker matrix
    initializer = initializers.random_matrix(((2, 3), (3, 2)), tt_rank=1)
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    self.assertTrue(kr._is_kron(kron_mat))

  def testDet(self):
    # Tests the determinant function
    initializer = initializers.random_matrix(((2, 3, 2), (2, 3, 2)), tt_rank=1)
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      desired = np.linalg.det(ops.full(kron_mat).eval())
      actual = kr.determinant(kron_mat).eval()
      self.assertAllClose(desired, actual)

  def testSlogDet(self):
    # Tests the slog_determinant function
    
    # TODO: use kron and -1 * kron matrices, when mul is implemented 
    # the current version is platform-dependent
    
    tf.set_random_seed(5) # negative derminant
    initializer = initializers.random_matrix(((2, 3), (2, 3)), tt_rank=1)
    kron_neg = variables.get_variable('kron_neg', initializer=initializer)
  
    tf.set_random_seed(1) # positive determinant
    initializer = initializers.random_matrix(((2, 3), (2, 3)), tt_rank=1)
    kron_pos = variables.get_variable('kron_pos', initializer=initializer)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
       # negative derminant
      sess.run(init_op)
      desired_sign, desired_det = np.linalg.slogdet(ops.full(kron_neg).eval())
      actual_sign, actual_det = sess.run(kr.slog_determinant(kron_neg))
      self.assertEqual(desired_sign, actual_sign)
      self.assertAllClose(desired_det, actual_det)
 
      # positive determinant 
      desired_sign, desired_det = np.linalg.slogdet(ops.full(kron_pos).eval())
      actual_sign, actual_det = sess.run(kr.slog_determinant(kron_pos))
      self.assertEqual(desired_sign, actual_sign)
      self.assertAllClose(desired_det, actual_det)

  def testInv(self):
    # Tests the inv function
    initializer = initializers.random_matrix(((2, 3, 2), (2, 3, 2)), tt_rank=1)
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      desired = np.linalg.inv(ops.full(kron_mat).eval())
      actual = ops.full(kr.inv(kron_mat)).eval()
      self.assertAllClose(desired, actual)
    
  def testCholesky(self):
    # Tests the cholesky function
    np.random.seed(8)

    # generating two symmetric positive-definite tt-cores
    L_1 = np.tril(np.random.normal(scale=2., size=(2, 2)))
    L_2 = np.tril(np.random.normal(scale=2., size=(3, 3)))
    K_1 = L_1.dot(L_1.T)
    K_2 = L_2.dot(L_2.T)
    K = np.kron(K_1, K_2)
    initializer = tensor_train.TensorTrain([K_1[None, :, :, None], 
                                            K_2[None, :, :, None]], 
                                            tt_ranks=7*[1])
    kron_mat = variables.get_variable('kron_mat', initializer=initializer)
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      desired = np.linalg.cholesky(K)
      actual = ops.full(kr.cholesky(kron_mat)).eval()
      self.assertAllClose(desired, actual)


if __name__ == "__main__":
  tf.test.main()
