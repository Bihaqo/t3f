import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain
import ops
import initializers
import riemannian


class RiemannianTest(tf.test.TestCase):

  def testProjectOnItself(self):
    # Projection of X into the tangent space of itself is X: P_x(x) = x.
    tens = initializers.random_tensor((2, 3, 4))
    proj = riemannian.project(tens, tens)
    with self.test_session() as sess:
      actual_val, desired_val = sess.run((ops.full(proj), ops.full(tens)))
      self.assertAllClose(desired_val, actual_val)

  def testProject(self):
    # Compare our projection with the results obtained (and precomputed) from
    # tt.riemannian.project which is well tested.
    tangent_tens_cores = ([[[-0.42095269,  0.02130842],
         [-0.4181081 ,  0.42945687],
         [ 0.45972439, -0.4525616 ],
         [-0.17159869, -0.14505528]]], [[[ 0.23344421],
         [ 0.81480049],
         [-0.92385135]],

        [[-0.19279465],
         [ 0.524976  ],
         [-0.40149197]]])
    tangent_tens = TensorTrain(tangent_tens_cores, (4, 3), (1, 2, 1))
    tens_cores = ([[[-1.01761142,  0.36075896, -0.2493624 ],
         [-0.99896565, -1.12685474,  1.02832458],
         [ 1.08739724, -0.6537435 ,  1.99975537],
         [ 0.35128005,  0.40395104, -0.16790072]]], [[[ 0.34105142],
         [ 0.10371947],
         [-1.76464904]],

        [[ 0.32639758],
         [-1.27843174],
         [-1.41590327]],

        [[ 0.76616274],
         [ 0.6577514 ],
         [ 2.13703185]]])
    tens = TensorTrain(tens_cores, (4, 3), (1, 3, 1))
    desired_projection = [[-0.67638254, -1.17163914,  0.29850939],
       [-1.66479093, -0.99003251,  2.46629195],
       [-0.04847773, -0.72908174,  0.20142675],
       [ 0.34431125, -0.20935516, -1.15864246]]
    proj = riemannian.project(tangent_tens, tens)
    with self.test_session() as sess:
      self.assertAllClose(desired_projection, ops.full(proj).eval())

  def testProjectSum(self):
    # Test projecting a batch of TT-tensors.
    tens = initializers.random_tensor_batch((2, 3, 4), batch_size=3)
    tangent_tens = initializers.random_tensor((2, 3, 4), 3)
    weighted_sum = tens[0] + tens[1] + tens[2]
    direct_proj = riemannian.project(tangent_tens, weighted_sum)
    actual_proj = riemannian.project(tangent_tens, tens)
    with self.test_session() as sess:
      res = sess.run((ops.full(direct_proj), ops.full(actual_proj)))
      desired_val, actual_val = res
      self.assertAllClose(desired_val, actual_val)

  def testProjectSumCoef(self):
    # Test projecting a batch of TT-tensors with providing coefs.
    tens = initializers.random_tensor_batch((2, 3, 4), 3, batch_size=4)
    coef = [0.1, -2, 0, 0.4]
    tangent_tens = initializers.random_tensor((2, 3, 4), 4)
    weighted_sum = coef[0] * tens[0] + coef[1] * tens[1] + coef[2] * tens[2]
    weighted_sum += coef[3] * tens[3]
    direct_proj = riemannian.project(tangent_tens, weighted_sum)
    actual_proj = riemannian.project(tangent_tens, tens, coef)
    with self.test_session() as sess:
      res = sess.run((ops.full(direct_proj), ops.full(actual_proj)))
      desired_val, actual_val = res
      self.assertAllClose(desired_val, actual_val)

  def testProjectMatrixOnItself(self):
    # Project a TT-matrix on itself.
    # Projection of X into the tangent space of itself is X: P_x(x) = x.
    tt_mat = initializers.random_matrix(((2, 3, 4), (2, 3, 4)))
    proj = riemannian.project(tt_mat, tt_mat)
    with self.test_session() as sess:
      actual_val, desired_val = sess.run((ops.full(proj), ops.full(tt_mat)))
      self.assertAllClose(desired_val, actual_val)


if __name__ == "__main__":
  tf.test.main()
