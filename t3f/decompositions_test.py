import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from t3f import ops
from t3f import shapes
from t3f import decompositions
from t3f import initializers


class _DecompositionsTest():

  def testTTTensor(self):
    shape = (2, 1, 4, 3)
    np.random.seed(1)
    tens = np.random.rand(*shape).astype(self.dtype.as_numpy_dtype)
    tf_tens = tf.constant(tens)
    tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=3)
    self.assertAllClose(tens, self.evaluate(ops.full(tt_tens)))
    dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(tt_tens))
    static_tt_ranks = tt_tens.get_tt_ranks().as_list()
    self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)

  def testTTTensorSimple(self):
    # Test that a tensor of ones and of zeros can be converted into TT with
    # TT-rank 1.
    shape = (2, 1, 4, 3)
    tens_arr = (np.zeros(shape).astype(self.dtype.as_numpy_dtype),
                np.ones(shape).astype(self.dtype.as_numpy_dtype))
    for tens in tens_arr:
      tf_tens = tf.constant(tens)
      tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=1)
      self.assertAllClose(tens, self.evaluate(ops.full(tt_tens)))
      dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(tt_tens))
      static_tt_ranks = tt_tens.get_tt_ranks().as_list()
      self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)

  def testTTVector(self):
    vec_shape = (2, 1, 4, 3)
    np.random.seed(1)
    rows = np.prod(vec_shape)
    vec = np.random.rand(rows, 1).astype(self.dtype.as_numpy_dtype)
    tf_vec = tf.constant(vec)
    tt_vec = decompositions.to_tt_matrix(tf_vec, (vec_shape, None))
    self.assertAllClose(vec, self.evaluate(ops.full(tt_vec)))

  def testTTCompositeRankTensor(self):
    # Test if a composite rank (list of ranks) can be used for decomposition
    # for tensor.
    np.random.seed(1)
    np_tensor = np.random.rand(2, 3, 3, 1).astype(self.dtype.as_numpy_dtype)
    tf_tensor = tf.constant(np_tensor)

    tt_ranks = [1, 2, 3, 3, 1]
    tt_tensor = decompositions.to_tt_tensor(tf_tensor, max_tt_rank=tt_ranks)
    self.assertAllClose(np_tensor, self.evaluate(ops.full(tt_tensor)))

  def testTTCompositeRankMatrix(self):
    # Test if a composite rank (list of ranks) can be used for decomposition
    # for a matrix.
    inp_shape = (2, 3, 3, 2)
    out_shape = (1, 2, 2, 1)
    np.random.seed(1)
    mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape))
    mat = mat.astype(self.dtype.as_numpy_dtype)
    tf_mat = tf.constant(mat)
    tt_ranks = [10, 20, 30, 40, 30]
    tt_mat = decompositions.to_tt_matrix(tf_mat, (out_shape, inp_shape),
                                         max_tt_rank=tt_ranks)
    self.assertAllClose(mat, self.evaluate(ops.full(tt_mat)), atol=1e-5, rtol=1e-5)

  def testTTMatrix(self):
    # Convert a np.prod(out_shape) x np.prod(in_shape) matrix into TT-matrix
    # and back.
    inp_shape = (2, 5, 2, 3)
    out_shape = (3, 3, 2, 3)
    np.random.seed(1)
    mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape))
    mat = mat.astype(self.dtype.as_numpy_dtype)
    tf_mat = tf.constant(mat)
    tt_mat = decompositions.to_tt_matrix(tf_mat, (out_shape, inp_shape),
                                         max_tt_rank=90)
    # TODO: why so bad accuracy?
    self.assertAllClose(mat, self.evaluate(ops.full(tt_mat)), atol=1e-5, rtol=1e-5)

  def testRoundTensor(self):
    shape = (2, 1, 4, 3, 3)
    np.random.seed(1)
    tens = initializers.random_tensor(shape, tt_rank=15,
                                      dtype=self.dtype)
    rounded_tens = decompositions.round(tens, max_tt_rank=9)
    vars = [ops.full(tens), ops.full(rounded_tens)]
    tens_value, rounded_tens_value = self.evaluate(vars)
    # TODO: why so bad accuracy?
    self.assertAllClose(tens_value, rounded_tens_value, atol=1e-4, rtol=1e-4)
    dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(rounded_tens))
    self.assertAllEqual([1, 2, 2, 8, 3, 1], dynamic_tt_ranks)

  def testOrthogonalizeLeftToRight(self):
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 2, 2, 6, 1)
    tens = initializers.random_tensor(shape, tt_rank=tt_ranks,
                                      dtype=self.dtype)
    orthogonal = decompositions.orthogonalize_tt_cores(tens)
    tens_val, orthogonal_val = self.evaluate([ops.full(tens), ops.full(orthogonal)])
    self.assertAllClose(tens_val, orthogonal_val, atol=1e-5, rtol=1e-5)
    dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(orthogonal))
    self.assertAllEqual(updated_tt_ranks, dynamic_tt_ranks)
    # Check that the TT-cores are orthogonal.
    for core_idx in range(4 - 1):
      core = orthogonal.tt_cores[core_idx]
      core = tf.reshape(core, (updated_tt_ranks[core_idx] * shape[core_idx],
                               updated_tt_ranks[core_idx + 1]))
      should_be_eye = tf.matmul(tf.transpose(core), core)
      should_be_eye_val = self.evaluate(should_be_eye)
      self.assertAllClose(np.eye(updated_tt_ranks[core_idx + 1]),
                          should_be_eye_val)

  def testOrthogonalizeRightToLeft(self):
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 5, 2, 3, 1)
    tens = initializers.random_tensor(shape, tt_rank=tt_ranks,
                                      dtype=self.dtype)
    orthogonal = decompositions.orthogonalize_tt_cores(tens, left_to_right=False)
    tens_val, orthogonal_val = self.evaluate([ops.full(tens), ops.full(orthogonal)])
    self.assertAllClose(tens_val, orthogonal_val, atol=1e-5, rtol=1e-5)
    dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(orthogonal))
    self.assertAllEqual(updated_tt_ranks, dynamic_tt_ranks)
    # Check that the TT-cores are orthogonal.
    for core_idx in range(1, 4):
      core = orthogonal.tt_cores[core_idx]
      core = tf.reshape(core, (updated_tt_ranks[core_idx], shape[core_idx] *
                               updated_tt_ranks[core_idx + 1]))
      should_be_eye = tf.matmul(core, tf.transpose(core))
      should_be_eye_val = self.evaluate(should_be_eye)
      self.assertAllClose(np.eye(updated_tt_ranks[core_idx]),
                          should_be_eye_val)


class _DecompositionsBatchTest():

  def testOrthogonalizeLeftToRight(self):
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 2, 2, 6, 1)
    tens = initializers.random_tensor_batch(shape, tt_rank=tt_ranks,
                                            batch_size=2, dtype=self.dtype)
    orthogonal = decompositions.orthogonalize_tt_cores(tens)
    tens_val, orthogonal_val = self.evaluate([ops.full(tens), ops.full(orthogonal)])
    self.assertAllClose(tens_val, orthogonal_val, atol=1e-5, rtol=1e-5)
    dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(orthogonal))
    self.assertAllEqual(updated_tt_ranks, dynamic_tt_ranks)
    # Check that the TT-cores are orthogonal.
    for core_idx in range(4 - 1):
      core_shape = (updated_tt_ranks[core_idx] * shape[core_idx],
                    updated_tt_ranks[core_idx + 1])
      for i in range(2):
        core = tf.reshape(orthogonal.tt_cores[core_idx][i], core_shape)
        should_be_eye = tf.matmul(tf.transpose(core), core)
        should_be_eye_val = self.evaluate(should_be_eye)
        self.assertAllClose(np.eye(updated_tt_ranks[core_idx + 1]),
                            should_be_eye_val)

  def testRoundTensor(self):
    shape = (2, 1, 4, 3, 3)
    tens = initializers.random_tensor_batch(shape, tt_rank=15, batch_size=3,
                                            dtype=self.dtype)
    rounded_tens = decompositions.round(tens, max_tt_rank=9)
    vars = [ops.full(tens), ops.full(rounded_tens)]
    tens_value, rounded_tens_value = self.evaluate(vars)
    # TODO: why so bad accuracy?
    self.assertAllClose(tens_value, rounded_tens_value, atol=1e-4,
                        rtol=1e-4)
    dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(rounded_tens))
    self.assertAllEqual([1, 2, 2, 8, 3, 1], dynamic_tt_ranks)


class DecompositionsTestFloat32(tf.test.TestCase, _DecompositionsTest):
  dtype = tf.float32


class DecompositionsTestFloat64(tf.test.TestCase, _DecompositionsTest):
  dtype = tf.float64


class DecompositionsBatchTestFloat32(tf.test.TestCase, _DecompositionsBatchTest):
  dtype = tf.float32


class DecompositionsBatchTestFloat64(tf.test.TestCase, _DecompositionsBatchTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
