import numpy as np
import tensorflow as tf

import ops
import shapes
import decompositions
import initializers


class DecompositionsTest(tf.test.TestCase):

  def testTTTensor(self):
    shape = (2, 1, 4, 3)
    np.random.seed(1)
    tens = np.random.rand(*shape).astype(np.float32)
    tf_tens = tf.constant(tens)
    tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=3)
    with self.test_session():
      self.assertAllClose(tens, ops.full(tt_tens).eval())
      dynamic_tt_ranks = shapes.tt_ranks(tt_tens).eval()
      static_tt_ranks = tt_tens.get_tt_ranks().as_list()
      self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)

      # Try to decompose the same tensor with unknown shape.
      tf_tens_pl = tf.placeholder(tf.float32, (None, None, 4, None))
      tt_tens = decompositions.to_tt_tensor(tf_tens_pl, max_tt_rank=3)
      tt_val = ops.full(tt_tens).eval({tf_tens_pl: tens})
      self.assertAllClose(tens, tt_val)
      dynamic_tt_ranks = shapes.tt_ranks(tt_tens).eval({tf_tens_pl: tens})
      self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)

  def testTTTensorSimple(self):
    # Test that a tensor of ones and of zeros can be converted into TT with
    # TT-rank 1.
    shape = (2, 1, 4, 3)
    tens_arr = (np.zeros(shape).astype(np.float32),
                np.ones(shape).astype(np.float32))
    for tens in tens_arr:
      tf_tens = tf.constant(tens)
      tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=1)
      with self.test_session():
        self.assertAllClose(tens, ops.full(tt_tens).eval())
        dynamic_tt_ranks = shapes.tt_ranks(tt_tens).eval()
        static_tt_ranks = tt_tens.get_tt_ranks().as_list()
        self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)

        # Try to decompose the same tensor with unknown shape.
        tf_tens_pl = tf.placeholder(tf.float32, (None, None, None, None))
        tt_tens = decompositions.to_tt_tensor(tf_tens_pl, max_tt_rank=1)
        tt_val = ops.full(tt_tens).eval({tf_tens_pl: tens})
        self.assertAllClose(tens, tt_val)
        dynamic_tt_ranks = shapes.tt_ranks(tt_tens).eval({tf_tens_pl: tens})
        self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)

  def testTTVector(self):
    vec_shape = (2, 1, 4, 3)
    np.random.seed(1)
    rows = np.prod(vec_shape)
    vec = np.random.rand(rows, 1).astype(np.float32)
    tf_vec = tf.constant(vec)
    tt_vec = decompositions.to_tt_matrix(tf_vec, (vec_shape, None))
    with self.test_session():
      self.assertAllClose(vec, ops.full(tt_vec).eval())

  def testTTMatrix(self):
    # Convert a np.prod(out_shape) x np.prod(in_shape) matrix into TT-matrix
    # and back.
    inp_shape = (2, 5, 2, 3)
    out_shape = (3, 3, 2, 3)
    np.random.seed(1)
    mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape))
    mat = mat.astype(np.float32)
    tf_mat = tf.constant(mat)
    tt_mat = decompositions.to_tt_matrix(tf_mat, (out_shape, inp_shape),
                                         max_tt_rank=90)
    with self.test_session():
      # TODO: why so bad accuracy?
      self.assertAllClose(mat, ops.full(tt_mat).eval(), atol=1e-5, rtol=1e-5)

  def testRoundTensor(self):
    shape = (2, 1, 4, 3, 3)
    np.random.seed(1)
    tens = initializers.random_tensor(shape, tt_rank=15)
    rounded_tens = decompositions.round(tens, max_tt_rank=9)
    with self.test_session() as sess:
      vars = [ops.full(tens), ops.full(rounded_tens)]
      tens_value, rounded_tens_value = sess.run(vars)
      # TODO: why so bad accuracy?
      self.assertAllClose(tens_value, rounded_tens_value, atol=1e-4, rtol=1e-4)
      dynamic_tt_ranks = shapes.tt_ranks(rounded_tens).eval()
      self.assertAllEqual([1, 2, 2, 8, 3, 1], dynamic_tt_ranks)

  def testOrthogonalizeLeftToRight(self):
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 2, 2, 6, 1)
    tens = initializers.random_tensor(shape, tt_rank=tt_ranks)
    orthogonal = decompositions.orthogonalize_tt_cores(tens)
    with self.test_session() as sess:
      tens_val, orthogonal_val = sess.run([ops.full(tens), ops.full(orthogonal)])
      self.assertAllClose(tens_val, orthogonal_val, atol=1e-5, rtol=1e-5)
      dynamic_tt_ranks = shapes.tt_ranks(orthogonal).eval()
      self.assertAllEqual(updated_tt_ranks, dynamic_tt_ranks)
      # Check that the TT-cores are orthogonal.
      for core_idx in range(4 - 1):
        core = orthogonal.tt_cores[core_idx]
        core = tf.reshape(core, (updated_tt_ranks[core_idx] * shape[core_idx],
                                 updated_tt_ranks[core_idx + 1]))
        should_be_eye = tf.matmul(tf.transpose(core), core)
        should_be_eye_val = sess.run(should_be_eye)
        self.assertAllClose(np.eye(updated_tt_ranks[core_idx + 1]),
                            should_be_eye_val)

  def testOrthogonalizeRightToLeft(self):
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 5, 2, 3, 1)
    tens = initializers.random_tensor(shape, tt_rank=tt_ranks)
    orthogonal = decompositions.orthogonalize_tt_cores(tens, left_to_right=False)
    with self.test_session() as sess:
      tens_val, orthogonal_val = sess.run([ops.full(tens), ops.full(orthogonal)])
      self.assertAllClose(tens_val, orthogonal_val, atol=1e-5, rtol=1e-5)
      dynamic_tt_ranks = shapes.tt_ranks(orthogonal).eval()
      self.assertAllEqual(updated_tt_ranks, dynamic_tt_ranks)
      # Check that the TT-cores are orthogonal.
      for core_idx in range(1, 4):
        core = orthogonal.tt_cores[core_idx]
        core = tf.reshape(core, (updated_tt_ranks[core_idx], shape[core_idx] *
                                 updated_tt_ranks[core_idx + 1]))
        should_be_eye = tf.matmul(core, tf.transpose(core))
        should_be_eye_val = sess.run(should_be_eye)
        self.assertAllClose(np.eye(updated_tt_ranks[core_idx]),
                            should_be_eye_val)


class DecompositionsBatchTest(tf.test.TestCase):

  def testOrthogonalizeLeftToRight(self):
    shape = (2, 4, 3, 3)
    tt_ranks = (1, 5, 2, 17, 1)
    updated_tt_ranks = (1, 2, 2, 6, 1)
    tens = initializers.random_tensor_batch(shape, tt_rank=tt_ranks,
                                            batch_size=2)
    orthogonal = decompositions.orthogonalize_tt_cores(tens)
    with self.test_session() as sess:
      tens_val, orthogonal_val = sess.run([ops.full(tens), ops.full(orthogonal)])
      self.assertAllClose(tens_val, orthogonal_val, atol=1e-5, rtol=1e-5)
      dynamic_tt_ranks = shapes.tt_ranks(orthogonal).eval()
      self.assertAllEqual(updated_tt_ranks, dynamic_tt_ranks)
      # Check that the TT-cores are orthogonal.
      for core_idx in range(4 - 1):
        core_shape = (updated_tt_ranks[core_idx] * shape[core_idx],
                      updated_tt_ranks[core_idx + 1])
        for i in range(2):
          core = tf.reshape(orthogonal.tt_cores[core_idx][i], core_shape)
          should_be_eye = tf.matmul(tf.transpose(core), core)
          should_be_eye_val = sess.run(should_be_eye)
          self.assertAllClose(np.eye(updated_tt_ranks[core_idx + 1]),
                              should_be_eye_val)

  def testRoundTensor(self):
    shape = (2, 1, 4, 3, 3)
    tens = initializers.random_tensor_batch(shape, tt_rank=15, batch_size=3)
    rounded_tens = decompositions.round(tens, max_tt_rank=9)
    with self.test_session() as sess:
      vars = [ops.full(tens), ops.full(rounded_tens)]
      tens_value, rounded_tens_value = sess.run(vars)
      # TODO: why so bad accuracy?
      self.assertAllClose(tens_value, rounded_tens_value, atol=1e-4,
                          rtol=1e-4)
      dynamic_tt_ranks = shapes.tt_ranks(rounded_tens).eval()
      self.assertAllEqual([1, 2, 2, 8, 3, 1], dynamic_tt_ranks)


if __name__ == "__main__":
  tf.test.main()
