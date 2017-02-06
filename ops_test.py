import numpy as np
import tensorflow as tf

import tensor_train
import ops
import initializers


class TTTensorTest(tf.test.TestCase):

  def testFullTensor2d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(10, rank).astype(np.float32)
      b = np.random.rand(rank, 9).astype(np.float32)
      tt_cores = (a.reshape(1, 10, rank), b.reshape(rank, 9, 1))
      desired = np.dot(a, b)
      with self.test_session():
        tf_tens = tensor_train.TensorTrain(tt_cores)
        actual = ops.full(tf_tens)
        self.assertAllClose(desired, actual.eval())

  def testFullTensor3d(self):
    np.random.seed(1)
    for rank_1 in [1, 2]:
      a = np.random.rand(10, rank_1).astype(np.float32)
      b = np.random.rand(rank_1, 9, 3).astype(np.float32)
      c = np.random.rand(3, 8).astype(np.float32)
      tt_cores = (a.reshape(1, 10, rank_1), b, c.reshape((3, 8, 1)))
      # Basically do full by hand.
      desired = a.dot(b.reshape((rank_1, -1)))
      desired = desired.reshape((-1, 3)).dot(c)
      desired = desired.reshape(10, 9, 8)
      with self.test_session():
        tf_tens = tensor_train.TensorTrain(tt_cores)
        actual = ops.full(tf_tens)
        self.assertAllClose(desired, actual.eval())

  def testTTTensor(self):
    shape = (2, 1, 4, 3)
    np.random.seed(1)
    tens = np.random.rand(*shape).astype(np.float32)
    with self.test_session():
      tf_tens = tf.constant(tens)
      tt_tens = ops.to_tt_tensor(tf_tens, shape)
      self.assertAllClose(tens, ops.full(tt_tens).eval())


class TTMatrixTest(tf.test.TestCase):

  def testFullMatrix2d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(2, 3, rank).astype(np.float32)
      b = np.random.rand(rank, 4, 5).astype(np.float32)
      tt_cores = (a.reshape(1, 2, 3, rank), b.reshape((rank, 4, 5, 1)))
      # Basically do full by hand.
      desired = a.reshape((-1, rank)).dot(b.reshape((rank, -1)))
      desired = desired.reshape((2, 3, 4, 5))
      desired = desired.transpose((0, 2, 1, 3))
      desired = desired.reshape((2 * 4, 3 * 5))
      with self.test_session():
        tf_mat = tensor_train.TensorTrain(tt_cores)
        actual = ops.full(tf_mat)
        self.assertAllClose(desired, actual.eval())

  def testFullMatrix3d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(2, 3, rank).astype(np.float32)
      b = np.random.rand(rank, 4, 5, rank).astype(np.float32)
      c = np.random.rand(rank, 2, 2).astype(np.float32)
      tt_cores = (a.reshape(1, 2, 3, rank), b.reshape(rank, 4, 5, rank),
                  c.reshape(rank, 2, 2, 1))
      # Basically do full by hand.
      desired = a.reshape((-1, rank)).dot(b.reshape((rank, -1)))
      desired = desired.reshape((-1, rank)).dot(c.reshape((rank, -1)))
      desired = desired.reshape((2, 3, 4, 5, 2, 2))
      desired = desired.transpose((0, 2, 4, 1, 3, 5))
      desired = desired.reshape((2 * 4 * 2, 3 * 5 * 2))
      with self.test_session():
        tf_mat = tensor_train.TensorTrain(tt_cores)
        actual = ops.full(tf_mat)
        self.assertAllClose(desired, actual.eval())

  def testTTVector(self):
    vec_shape = (2, 1, 4, 3)
    np.random.seed(1)
    vec = np.random.rand(np.prod(vec_shape)).astype(np.float32)
    with self.test_session():
      tf_vec = tf.constant(vec)
      tt_vec = ops.to_tt_matrix(tf_vec, vec_shape)
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
      tt_mat = ops.to_tt_matrix(tf_mat, (out_shape, inp_shape))
      self.assertAllClose(mat, ops.full(tt_mat).eval())

  def testTTMatTimesTTMat(self):
    # Multiply a TT-matrix by another TT-matrix.
    left_shape = (2, 3, 4)
    sum_shape = (4, 3, 5)
    right_shape = (4, 4, 4)
    with self.test_session() as sess:
      tt_mat_1 = initializers.tt_rand_matrix((left_shape, sum_shape), tt_rank=3)
      tt_mat_2 = initializers.tt_rand_matrix((sum_shape, right_shape))
      res_actual = ops.tt_tt_matmul(tt_mat_1, tt_mat_2)
      res_actual = ops.full(res_actual)
      res_desired = tf.matmul(ops.full(tt_mat_1), ops.full(tt_mat_2))
      res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
      # TODO: why so bad accuracy?
      self.assertAllClose(res_actual_val, res_desired_val, atol=1e-4, rtol=1e-4)

  def testTTMatTimesDenseVec(self):
    # Multiply a TT-matrix by a dense vector.
    inp_shape = (2, 3, 4)
    out_shape = (3, 4, 3)
    np.random.seed(1)
    vec = np.random.rand(np.prod(inp_shape), 1).astype(np.float32)
    with self.test_session():
      tf_vec = tf.constant(vec)
      tf.set_random_seed(1)
      tt_mat = initializers.tt_rand_matrix((out_shape, inp_shape))
      res_actual = ops.matmul(tt_mat, tf_vec)
      res_desired = tf.matmul(ops.full(tt_mat), tf_vec)
      self.assertAllClose(res_actual.eval(), res_desired.eval())

  def testDenseMatTimesTTVec(self):
    # Multiply a TT-matrix by a dense vector.
    inp_shape = (3, 3, 3, 3)
    out_shape = (3, 3, 3, 3)
    np.random.seed(1)
    mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape)).astype(np.float32)
    with self.test_session():
      tf_mat = tf.constant(mat)
      tf.set_random_seed(1)
      tt_vec = initializers.tt_rand_matrix((inp_shape, None))
      res_actual = ops.matmul(tf_mat, tt_vec)
      res_desired = tf.matmul(ops.full(tf_mat), tt_vec)
      self.assertAllClose(res_actual.eval(), res_desired.eval())

  def testFlatInnerTTTensbyTTTens(self):
    # Inner product between two TT-tensors.
    shape_list = ((2, 2),
                  (2, 3, 4),
                  (4, 2, 5, 2))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for shape in shape_list:
        for rank in rank_list:
          tt_1 = initializers.tt_rand_tensor(shape, tt_rank=rank)
          tt_2 = initializers.tt_rand_tensor(shape, tt_rank=rank)
          res_actual = ops.tt_tt_flat_inner(tt_1, tt_2)
          tt_1_full = tf.reshape(ops.full(tt_1), (1, -1))
          tt_2_full = tf.reshape(ops.full(tt_2), (-1, 1))
          res_desired = tf.matmul(tt_1_full, tt_2_full)
          res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
          self.assertAllClose(res_actual_val, res_desired_val)

  def testFlatInnerTTMatbyTTMat(self):
    # Inner product between two TT-Matrices.
    shape_list = (((2, 2), (3, 4)),
                  ((2, 3, 4), (2, 2, 2)))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for shape in shape_list:
        for rank in rank_list:
          tt_1 = initializers.tt_rand_matrix(shape, tt_rank=rank)
          tt_2 = initializers.tt_rand_matrix(shape, tt_rank=rank)
          res_actual = ops.tt_tt_flat_inner(tt_1, tt_2)
          tt_1_full = tf.reshape(ops.full(tt_1), (1, -1))
          tt_2_full = tf.reshape(ops.full(tt_2), (-1, 1))
          res_desired = tf.matmul(tt_1_full, tt_2_full)
          res_actual_val, res_desired_val = sess.run(
            [res_actual, res_desired])
          self.assertAllClose(res_actual_val, res_desired_val, rtol=1e-5, atol=1e-5)


  def testFlatInnerTTTensbySparseTens(self):
    # Inner product between a TT-tensor and a sparse tensor.
    shape_list = ((2, 2),
                  (2, 3, 4),
                  (4, 2, 5, 2))
    rank_list = (1, 2)
    np.random.seed(1)
    with self.test_session() as sess:
      for shape in shape_list:
        for rank in rank_list:
          tt_1 = initializers.tt_rand_tensor(shape, tt_rank=rank)
          # TODO: test with sparse tensor with one element.
          sparse_flat_indices = np.random.choice(np.prod(shape), 20).astype(int)
          sparse_indices = np.unravel_index(sparse_flat_indices, shape)
          sparse_indices = np.vstack(sparse_indices).transpose()
          values = np.random.randn(20).astype(np.float32)
          sparse_2 = tf.SparseTensor(indices=sparse_indices, values=values,
                                     shape=shape)
          res_actual = ops.tt_sparse_flat_inner(tt_1, sparse_2)
          res_actual_val, tt_1_val = sess.run([res_actual, ops.full(tt_1)])
          res_desired_val = tt_1_val.flatten()[sparse_flat_indices].dot(values)
          self.assertAllClose(res_actual_val, res_desired_val)

if __name__ == "__main__":
  tf.test.main()
