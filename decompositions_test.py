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

  def testTTVector(self):
    vec_shape = (2, 1, 4, 3)
    np.random.seed(1)
    rows = np.prod(vec_shape)
    vec = np.random.rand(rows, 1).astype(np.float32)
    with self.test_session():
      tf_vec = tf.constant(vec)
      tt_vec = decompositions.to_tt_matrix(tf_vec, (vec_shape, None))
      self.assertAllClose(vec, ops.full(tt_vec).eval())

  def testTTMatrix(self):
    # Convert a np.prod(out_shape) x np.prod(in_shape) matrix into TT-matrix
    # and back.
    inp_shape = (2, 5, 2, 3)
    out_shape = (3, 3, 2, 3)
    np.random.seed(1)
    mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape)).astype(np.float32)
    with self.test_session():
      tf_mat = tf.constant(mat)
      tt_mat = decompositions.to_tt_matrix(tf_mat, (out_shape, inp_shape), max_tt_rank=90)
      # TODO: why so bad accuracy?
      self.assertAllClose(mat, ops.full(tt_mat).eval(), atol=1e-5, rtol=1e-5)

  def testRoundTensor(self):
    shape = (2, 1, 4, 3, 3)
    np.random.seed(1)
    tens = initializers.tt_rand_tensor(shape, tt_rank=10)
    rounded_tens = decompositions.round(tens, max_tt_rank=4)
    with self.test_session() as sess:
      vars = [ops.full(tens), ops.full(rounded_tens)]
      tens_value, rounded_tens_value = sess.run(vars)
      self.assertAllClose(tens_value, rounded_tens_value, atol=1e-5, rtol=1e-5)
      dynamic_tt_ranks = shapes.tt_ranks(rounded_tens).eval()
      # The ranks shrinked because of orthogonalization.
      self.assertAllEqual([1, 2, 2, 8, 3, 1], dynamic_tt_ranks)

if __name__ == "__main__":
  tf.test.main()
