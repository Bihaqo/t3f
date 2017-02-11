import numpy as np
import tensorflow as tf

import t3f
import tensor_train
import kronecker as kr
import ops

class KroneckerTest(tf.test.TestCase):

  def testIsKronNonKron(self):
    """
    Tests _is_kron on a non-Kronecker matrix
    """
    tt_mat = t3f.get_variable('tt_mat', initializer=t3f.random_matrix(((2, 3), (3, 2)), tt_rank=2))
    self.assertFalse(kr._is_kron(tt_mat))
           
  def testIsKronKron(self):
    """
    Tests _is_kron on a Kronecker matrix
    """
    kron_mat = t3f.get_variable('kron_mat', initializer=t3f.random_matrix(((2, 3), (3, 2)), tt_rank=1))
    self.assertTrue(kr._is_kron(kron_mat))

  def testDet(self):
    """
    Tests the determinant function
    """
    tf.set_random_seed(5)
    kron_mat = t3f.get_variable('kron_mat', initializer=t3f.random_matrix(((2, 3, 2), (2, 3, 2)), tt_rank=1))
    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      desired = np.linalg.det(ops.full(kron_mat).eval())
      actual = kr.determinant(kron_mat)
      self.assertAllClose(desired, actual.eval())

  def testSlogDet(self):
    """
    Tests the slog_determinant function
    """
    tf.set_random_seed(5) # negative derminant
    kron_neg = t3f.get_variable('kron_neg', initializer=t3f.random_matrix(((2, 3), (2, 3)), tt_rank=1))
  
    tf.set_random_seed(1) # positive determinant
    kron_pos = t3f.get_variable('kron_pos', initializer=t3f.random_matrix(((2, 3), (2, 3)), tt_rank=1))

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


if __name__ == "__main__":
  tf.test.main()