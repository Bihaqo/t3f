import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.compat.v1.enable_resource_variables()

from t3f.tensor_train import TensorTrain
from t3f import ops
from t3f import initializers
from t3f import riemannian
from t3f import variables
from t3f import shapes
from t3f import batch_ops


class _RiemannianTest():

  def testProjectOnItself(self):
    # Projection of X into the tangent space of itself is X: P_x(x) = x.
    tens = initializers.random_tensor((2, 3, 4), dtype=self.dtype)
    proj = riemannian.project_sum(tens, tens)
    actual_val, desired_val = self.evaluate((ops.full(proj), ops.full(tens)))
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
    convert = lambda t: np.array(t, dtype=self.dtype.as_numpy_dtype)
    tangent_tens_cores = list([convert(t) for t in tangent_tens_cores])
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
    tens_cores = list([convert(t) for t in tens_cores])
    tens = TensorTrain(tens_cores, (4, 3), (1, 3, 1))
    desired_projection = [[-0.67638254, -1.17163914,  0.29850939],
       [-1.66479093, -0.99003251,  2.46629195],
       [-0.04847773, -0.72908174,  0.20142675],
       [ 0.34431125, -0.20935516, -1.15864246]]
    proj = riemannian.project_sum(tens, tangent_tens)
    proj_full = ops.full(proj)
    proj_v = self.evaluate(proj_full)
    self.assertAllClose(desired_projection, proj_v)
    self.assertEqual(self.dtype.as_numpy_dtype, proj_v.dtype)

  def testProjectSum(self):
    # Test projecting a batch of TT-tensors.
    tens = initializers.random_tensor_batch((2, 3, 4), batch_size=3,
                                            dtype=self.dtype)
    tangent_tens = initializers.random_tensor((2, 3, 4), 3,
                                              dtype=self.dtype)
    weighted_sum = tens[0] + tens[1] + tens[2]
    direct_proj = riemannian.project_sum(weighted_sum, tangent_tens)
    actual_proj = riemannian.project_sum(tens, tangent_tens)
    res = self.evaluate((ops.full(direct_proj), ops.full(actual_proj)))
    desired_val, actual_val = res
    self.assertAllClose(desired_val, actual_val)

  def testProjectWeightedSum(self):
    # Test projecting a batch of TT-tensors with providing coefs.
    tens = initializers.random_tensor_batch((2, 3, 4), 3, batch_size=4,
                                            dtype=self.dtype)
    coef = [0.1, -2, 0, 0.4]
    tangent_tens = initializers.random_tensor((2, 3, 4), 4,
                                              dtype=self.dtype)
    weighted_sum = coef[0] * tens[0] + coef[1] * tens[1] + coef[2] * tens[2]
    weighted_sum += coef[3] * tens[3]
    direct_proj = riemannian.project_sum(weighted_sum, tangent_tens)
    actual_proj = riemannian.project_sum(tens, tangent_tens, coef)
    res = self.evaluate((ops.full(direct_proj), ops.full(actual_proj)))
    desired_val, actual_val = res
    self.assertAllClose(desired_val, actual_val)

  def testProjectWeightedSumMultipleOutputs(self):
    # Test projecting a batch of TT-tensors with providing weights and outputing
    # several TT objects with different weights.
    tens = initializers.random_tensor_batch((2, 3, 4), 3, batch_size=4,
                                            dtype=self.dtype)
    np.random.seed(0)
    weights = np.random.randn(4, 2)
    tangent_tens = initializers.random_tensor((2, 3, 4), 4,
                                              dtype=self.dtype)
    weighted_sum_1 = weights[0, 0] * tens[0] + weights[1, 0] * tens[1] +\
                     weights[2, 0] * tens[2] + weights[3, 0] * tens[3]
    weighted_sum_2 = weights[0, 1] * tens[0] + weights[1, 1] * tens[1] +\
                     weights[2, 1] * tens[2] + weights[3, 1] * tens[3]
    direct_proj_1 = riemannian.project_sum(weighted_sum_1, tangent_tens)
    direct_proj_2 = riemannian.project_sum(weighted_sum_2, tangent_tens)
    direct_proj_1 = shapes.expand_batch_dim(direct_proj_1)
    direct_proj_2 = shapes.expand_batch_dim(direct_proj_2)
    direct_projs = batch_ops.concat_along_batch_dim((direct_proj_1, direct_proj_2))
    actual_proj = riemannian.project_sum(tens, tangent_tens, weights)
    res = self.evaluate((ops.full(direct_projs), ops.full(actual_proj)))
    desired_val, actual_val = res
    self.assertAllClose(desired_val, actual_val, atol=1e-5, rtol=1e-5)

  def testProjectWeightedSumDtypeBug(self):
    # Test that project_sum(TensorTrain, TensorTrain variable, np.array) works.
    what = initializers.random_tensor_batch((2, 3, 4), batch_size=3,
                                            dtype=self.dtype)
    where = variables.get_variable('a', initializer=what[0])
    weights = tf.zeros((3,), dtype=self.dtype)
    # Check that it doesn't throw an exception trying to convert weights to 
    # Variable dtype (float32_ref).
    riemannian.project_sum(what, where, weights)

  def testProjectMatrixOnItself(self):
    # Project a TT-matrix on itself.
    # Projection of X into the tangent space of itself is X: P_x(x) = x.
    tt_mat = initializers.random_matrix(((2, 3, 4), (2, 3, 4)),
                                        dtype=self.dtype)
    proj = riemannian.project_sum(tt_mat, tt_mat)
    actual_val, desired_val = self.evaluate((ops.full(proj), ops.full(tt_mat)))
    self.assertAllClose(desired_val, actual_val)

  def testCompareProjectSumAndProject(self):
    # Compare results of project_sum and project.
    tens = initializers.random_tensor_batch((2, 3, 4), 3, batch_size=4,
                                            dtype=self.dtype)
    tangent_tens = initializers.random_tensor((2, 3, 4), 4,
                                              dtype=self.dtype)
    project_sum = riemannian.project_sum(tens, tangent_tens, np.eye(4))
    project = riemannian.project(tens, tangent_tens)
    res = self.evaluate((ops.full(project_sum), ops.full(project)))
    project_sum_val, project_val = res
    self.assertAllClose(project_sum_val, project_val)

  def testProjectMatmul(self):
    # Project a TT-matrix times TT-vector on a TT-vector.
    tt_mat = initializers.random_matrix(((2, 3, 4), (2, 3, 4)),
                                        dtype=self.dtype)
    tt_vec_what = initializers.random_matrix_batch(((2, 3, 4), None),
                                                   batch_size=3,
                                                   dtype=self.dtype)
    tt_vec_where = initializers.random_matrix(((2, 3, 4), None),
                                              dtype=self.dtype)
    proj = riemannian.project_matmul(tt_vec_what, tt_vec_where, tt_mat)
    matvec = ops.matmul(tt_mat, tt_vec_what)
    proj_desired = riemannian.project(matvec, tt_vec_where)
    actual_val, desired_val = self.evaluate((ops.full(proj), ops.full(proj_desired)))
    self.assertAllClose(desired_val, actual_val, atol=1e-5, rtol=1e-5)

  def testPairwiseFlatInnerTensor(self):
    # Compare pairwise_flat_inner_projected against naive implementation.
    what1 = initializers.random_tensor_batch((2, 3, 4), 4, batch_size=3,
                                             dtype=self.dtype)
    what2 = initializers.random_tensor_batch((2, 3, 4), 4, batch_size=4,
                                             dtype=self.dtype)
    where = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    projected1 = riemannian.project(what1, where)
    projected2 = riemannian.project(what2, where)
    desired = batch_ops.pairwise_flat_inner(projected1, projected2)
    actual = riemannian.pairwise_flat_inner_projected(projected1, projected2)
    desired_val, actual_val = self.evaluate((desired, actual))
    self.assertAllClose(desired_val, actual_val, atol=1e-5, rtol=1e-5)

    with self.assertRaises(ValueError):
      # Second argument is not a projection on the tangent space.
      riemannian.pairwise_flat_inner_projected(projected1, what2)
    where2 = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    another_projected2 = riemannian.project(what2, where2)
    with self.assertRaises(ValueError):
      # The arguments are projections on different tangent spaces.
      riemannian.pairwise_flat_inner_projected(projected1, another_projected2)

  def testPairwiseFlatInnerMatrix(self):
    # Compare pairwise_flat_inner_projected against naive implementation.
    what1 = initializers.random_matrix_batch(((2, 3, 4), None), 4, batch_size=3,
                                             dtype=self.dtype)
    what2 = initializers.random_matrix_batch(((2, 3, 4), None), 4, batch_size=4,
                                             dtype=self.dtype)
    where = initializers.random_matrix(((2, 3, 4), None), 3,
                                       dtype=self.dtype)
    projected1 = riemannian.project(what1, where)
    projected2 = riemannian.project(what2, where)
    desired = batch_ops.pairwise_flat_inner(projected1, projected2)
    actual = riemannian.pairwise_flat_inner_projected(projected1, projected2)
    desired_val, actual_val = self.evaluate((desired, actual))
    self.assertAllClose(desired_val, actual_val, atol=1e-5, rtol=1e-5)

    with self.assertRaises(ValueError):
      # Second argument is not a projection on the tangent space.
      riemannian.pairwise_flat_inner_projected(projected1, what2)
    where2 = initializers.random_matrix(((2, 3, 4), None), 3,
                                        dtype=self.dtype)
    another_projected2 = riemannian.project(what2, where2)
    with self.assertRaises(ValueError):
      # The arguments are projections on different tangent spaces.
      riemannian.pairwise_flat_inner_projected(projected1, another_projected2)

  def testAddNProjected(self):
    # Add several TT-objects from the same tangent space.
    what1 = initializers.random_tensor_batch((2, 3, 4), 4, batch_size=3,
                                             dtype=self.dtype)
    what2 = initializers.random_tensor_batch((2, 3, 4), 3, batch_size=3,
                                             dtype=self.dtype)
    where = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    projected1 = riemannian.project(what1, where)
    projected2 = riemannian.project(what2, where)
    desired = ops.full(projected1 + projected2)
    actual = ops.full(riemannian.add_n_projected((projected1, projected2)))
    desired_val, actual_val = self.evaluate((desired, actual))
    self.assertAllClose(desired_val, actual_val, atol=1e-5, rtol=1e-5)

    with self.assertRaises(ValueError):
      # Second argument is not a projection on the tangent space.
      riemannian.add_n_projected((projected1, what2))
    where2 = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    another_projected2 = riemannian.project(what2, where2)
    with self.assertRaises(ValueError):
      # The arguments are projections on different tangent spaces.
      riemannian.add_n_projected((projected1, another_projected2))

  def testWeightedAddNProjected(self):
    # Add several TT-objects from the same tangent space with coefs.
    what1 = initializers.random_tensor((2, 3, 4), 4, dtype=self.dtype)
    what2 = initializers.random_tensor((2, 3, 4), 1, dtype=self.dtype)
    where = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    projected1 = riemannian.project(what1, where)
    projected2 = riemannian.project(what2, where)
    desired = ops.full(1.2 * projected1 + -2.0 * projected2)
    actual = ops.full(riemannian.add_n_projected((projected1, projected2),
                                                 coef=[1.2, -2.0]))
    desired_val, actual_val = self.evaluate((desired, actual))
    self.assertAllClose(desired_val, actual_val)

    with self.assertRaises(ValueError):
      # Second argument is not a projection on the tangent space.
      riemannian.add_n_projected((projected1, what2), coef=[1.2, -2.0])
    where2 = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    another_projected2 = riemannian.project(what2, where2)
    with self.assertRaises(ValueError):
      # The arguments are projections on different tangent spaces.
      riemannian.add_n_projected((projected1, another_projected2),
                                 coef=[1.2, -2.0])

  def testWeightedAddNProjectedBatch(self):
    # Add several TT-batches from the same tangent space with coefs.
    what1 = initializers.random_tensor_batch((2, 3, 4), 4, batch_size=3,
                                             dtype=self.dtype)
    what2 = initializers.random_tensor_batch((2, 3, 4), 1, batch_size=3,
                                             dtype=self.dtype)
    where = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    projected1 = riemannian.project(what1, where)
    projected2 = riemannian.project(what2, where)

    desired_0 = ops.full(1.2 * projected1[0] + -2.0 * projected2[0])
    desired_1 = ops.full(1.9 * projected1[1] + 2.0 * projected2[1])
    desired_2 = ops.full(0.0 * projected1[2] + 1.0 * projected2[2])
    desired = tf.stack((desired_0, desired_1, desired_2), axis=0)
    actual = ops.full(riemannian.add_n_projected((projected1, projected2),
                                                 coef=[[1.2, 1.9, 0.0],
                                                       [-2.0, 2.0, 1.0]]))
    desired_val, actual_val = self.evaluate((desired, actual))
    self.assertAllClose(desired_val, actual_val, atol=1e-5, rtol=1e-5)

  def testToAndFromDeltas(self):
    # Test converting to and from deltas representation of the tangent space
    # element.
    what = initializers.random_tensor((2, 3, 4), 4, dtype=self.dtype)
    where = initializers.random_tensor((2, 3, 4), 3, dtype=self.dtype)
    projected = riemannian.project(what, where)

    deltas = riemannian.tangent_space_to_deltas(projected)
    reconstructed_projected = riemannian.deltas_to_tangent_space(deltas, where)
    # Tangent space element norm can be computed from deltas norm.
    projected_normsq_desired = ops.frobenius_norm_squared(projected)
    projected_normsq_actual = tf.add_n([tf.reduce_sum(c * c) for c in deltas])
    desired_val, actual_val = self.evaluate((ops.full(projected),
                                        ops.full(reconstructed_projected)))
    self.assertAllClose(desired_val, actual_val)
    desired_val, actual_val = self.evaluate((projected_normsq_desired,
                                        projected_normsq_actual))
    self.assertAllClose(desired_val, actual_val)

  def testToAndFromDeltasBatch(self):
    # Test converting to and from deltas representation of the tangent space
    # element in the batch case.
    what = initializers.random_matrix_batch(((2, 3, 4), (3, 3, 3)), 4,
                                            batch_size=3, dtype=self.dtype)
    where = initializers.random_matrix(((2, 3, 4), (3, 3, 3)), 3,
                                       dtype=self.dtype)
    projected = riemannian.project(what, where)

    deltas = riemannian.tangent_space_to_deltas(projected)
    reconstructed_projected = riemannian.deltas_to_tangent_space(deltas, where)
    # Tangent space element norm can be computed from deltas norm.
    projected_normsq_desired = ops.frobenius_norm_squared(projected)
    d_normssq = [tf.reduce_sum(tf.reshape(c, (3, -1)) ** 2, 1) for c in deltas]
    projected_normsq_actual = tf.add_n(d_normssq)

    desired_val, actual_val = self.evaluate((ops.full(projected),
                                        ops.full(reconstructed_projected)))
    self.assertAllClose(desired_val, actual_val)
    desired_val, actual_val = self.evaluate((projected_normsq_desired,
                                        projected_normsq_actual))
    self.assertAllClose(desired_val, actual_val)


class RiemannianTestFloat32(tf.test.TestCase, _RiemannianTest):
  dtype = tf.float32


class RiemannianTestFloat64(tf.test.TestCase, _RiemannianTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
