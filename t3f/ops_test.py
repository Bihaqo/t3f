import numpy as np
import tensorflow as tf

from tensor_train import TensorTrain
from tensor_train_batch import TensorTrainBatch
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
        tf_tens = TensorTrain(tt_cores)
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
        tf_tens = TensorTrain(tt_cores)
        actual = ops.full(tf_tens)
        self.assertAllClose(desired, actual.eval())

  def testFlatInnerTTTensbyTTTens(self):
    # Inner product between two TT-tensors.
    shape_list = ((2, 2),
                  (2, 3, 4),
                  (4, 2, 5, 2))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for shape in shape_list:
        for rank in rank_list:
          tt_1 = initializers.random_tensor(shape, tt_rank=rank)
          tt_2 = initializers.random_tensor(shape, tt_rank=rank)
          res_actual = ops.flat_inner(tt_1, tt_2)
          tt_1_full = tf.reshape(ops.full(tt_1), (1, -1))
          tt_2_full = tf.reshape(ops.full(tt_2), (-1, 1))
          res_desired = tf.matmul(tt_1_full, tt_2_full)
          res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
          self.assertAllClose(res_actual_val, np.squeeze(res_desired_val),
                              rtol=1e-5)

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
          for num_elements in [1, 10]:
            tt_1 = initializers.random_tensor(shape, tt_rank=rank)
            sparse_flat_indices = np.random.choice(np.prod(shape), num_elements).astype(int)
            sparse_indices = np.unravel_index(sparse_flat_indices, shape)
            sparse_indices = np.vstack(sparse_indices).transpose()
            values = np.random.randn(num_elements).astype(np.float32)
            sparse_2 = tf.SparseTensor(indices=sparse_indices, values=values,
                                       dense_shape=shape)
            res_actual = ops.flat_inner(tt_1, sparse_2)
            res_actual_val, tt_1_val = sess.run([res_actual, ops.full(tt_1)])
            res_desired_val = tt_1_val.flatten()[sparse_flat_indices].dot(values)
            self.assertAllClose(res_actual_val, res_desired_val)

  def testAdd(self):
    # Sum two TT-tensors.
    tt_a = initializers.random_tensor((2, 1, 3, 4), tt_rank=2)
    tt_b = initializers.random_tensor((2, 1, 3, 4), tt_rank=[1, 2, 4, 3, 1])
    with self.test_session() as sess:
      res_actual = ops.full(ops.add(tt_a, tt_b))
      res_actual2 = ops.full(tt_a + tt_b)
      res_desired = ops.full(tt_a) + ops.full(tt_b)
      to_run = [res_actual, res_actual2, res_desired]
      res_actual_val, res_actual2_val, res_desired_val = sess.run(to_run)
      self.assertAllClose(res_actual_val, res_desired_val)
      self.assertAllClose(res_actual2_val, res_desired_val)

  def testMultiply(self):
    # Multiply two TT-tensors.
    tt_a = initializers.random_tensor((1, 2, 3, 4), tt_rank=2)
    tt_b = initializers.random_tensor((1, 2, 3, 4), tt_rank=[1, 1, 4, 3, 1])
    with self.test_session() as sess:
      res_actual = ops.full(ops.multiply(tt_a, tt_b))
      res_actual2 = ops.full(tt_a * tt_b)
      res_desired = ops.full(tt_a) * ops.full(tt_b)
      to_run = [res_actual, res_actual2, res_desired]
      res_actual_val, res_actual2_val, res_desired_val = sess.run(to_run)
      self.assertAllClose(res_actual_val, res_desired_val)
      self.assertAllClose(res_actual2_val, res_desired_val)

  def testFrobeniusNormTens(self):
    # Frobenius norm of a TT-tensor.
    shape_list = ((2, 2),
                  (2, 3, 4),
                  (4, 2, 5, 2))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for shape in shape_list:
        for rank in rank_list:
          tt = initializers.random_tensor(shape, tt_rank=rank)
          norm_sq_actual = ops.frobenius_norm_squared(tt)
          norm_actual = ops.frobenius_norm(tt)
          vars = [norm_sq_actual, norm_actual, ops.full(tt)]
          norm_sq_actual_val, norm_actual_val, tt_val = sess.run(vars)
          tt_val = tt_val.flatten()
          norm_sq_desired_val = tt_val.dot(tt_val)
          norm_desired_val = np.linalg.norm(tt_val)
          self.assertAllClose(norm_sq_actual_val, norm_sq_desired_val)
          self.assertAllClose(norm_actual_val, norm_desired_val, atol=1e-5,
                              rtol=1e-5)

  def testCastFloat(self):
    # Test cast function for float tt-tensors.
    tt_x = initializers.random_tensor((2, 3, 2), tt_rank=2)

    for dtype in [tf.float16, tf.float32, tf.float64]:
      self.assertEqual(ops.cast(tt_x, dtype).dtype, dtype)

  def testCastIntFloat(self):
    # Tests cast function from int to float for tensors.
    np.random.seed(1)
    K_1 = np.random.randint(0, high=100, size=(1, 2, 2))
    K_2 = np.random.randint(0, high=100, size=(2, 3, 2))
    K_3 = np.random.randint(0, high=100, size=(2, 2, 1))
    tt_int = TensorTrain([K_1, K_2, K_3], tt_ranks=[1, 2, 2, 1])
    
    for dtype in [tf.float16, tf.float32, tf.float64]:
      self.assertEqual(ops.cast(tt_int, dtype).dtype, dtype)


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
        tf_mat = TensorTrain(tt_cores)
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
        tf_mat = TensorTrain(tt_cores)
        actual = ops.full(tf_mat)
        self.assertAllClose(desired, actual.eval())

  def testTTMatTimesTTMat(self):
    # Multiply a TT-matrix by another TT-matrix.
    left_shape = (2, 3, 4)
    sum_shape = (4, 3, 5)
    right_shape = (4, 4, 4)
    with self.test_session() as sess:
      tt_mat_1 = initializers.random_matrix((left_shape, sum_shape), tt_rank=3)
      tt_mat_2 = initializers.random_matrix((sum_shape, right_shape))
      res_actual = ops.matmul(tt_mat_1, tt_mat_2)
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
    with self.test_session() as sess:
      tf_vec = tf.constant(vec)
      tf.set_random_seed(1)
      tt_mat = initializers.random_matrix((out_shape, inp_shape))
      res_actual = ops.matmul(tt_mat, tf_vec)
      res_desired = tf.matmul(ops.full(tt_mat), tf_vec)
      res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
      self.assertAllClose(res_actual_val, res_desired_val)

  def testDenseMatTimesTTVec(self):
    # Multiply a TT-matrix by a dense vector.
    inp_shape = (3, 3, 3, 3)
    out_shape = (3, 3, 3, 3)
    np.random.seed(1)
    mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape)).astype(np.float32)
    with self.test_session() as sess:
      tf_mat = tf.constant(mat)
      tf.set_random_seed(1)
      tt_vec = initializers.random_matrix((inp_shape, None))
      res_actual = ops.matmul(tf_mat, tt_vec)
      res_desired = tf.matmul(tf_mat, ops.full(tt_vec))
      res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
      self.assertAllClose(res_actual_val, res_desired_val, atol=1e-4, rtol=1e-4)

  def testFlatInnerTTMatbyTTMat(self):
    # Inner product between two TT-Matrices.
    shape_list = (((2, 2), (3, 4)),
                  ((2, 3, 4), (2, 2, 2)))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for shape in shape_list:
        for rank in rank_list:
          tt_1 = initializers.random_matrix(shape, tt_rank=rank)
          tt_2 = initializers.random_matrix(shape, tt_rank=rank)
          res_actual = ops.flat_inner(tt_1, tt_2)
          tt_1_full = tf.reshape(ops.full(tt_1), (1, -1))
          tt_2_full = tf.reshape(ops.full(tt_2), (-1, 1))
          res_desired = tf.matmul(tt_1_full, tt_2_full)
          res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
          self.assertAllClose(res_actual_val, np.squeeze(res_desired_val),
                              rtol=1e-5, atol=1e-5)

  def testFlatInnerTTMatbySparseMat(self):
    # Inner product between a TT-matrix and a sparse matrix.
    shape_list = (((2, 2), (3, 4)),
                  ((2, 3, 4), (2, 2, 2)))
    rank_list = (1, 2)
    np.random.seed(1)
    with self.test_session() as sess:
      for tensor_shape in shape_list:
        for rank in rank_list:
          for num_elements in [1, 9]:
            tt_1 = initializers.random_matrix(tensor_shape, tt_rank=rank)
            matrix_shape = np.prod(tensor_shape[0]), np.prod(tensor_shape[1])
            sparse_flat_indices = np.random.choice(np.prod(matrix_shape), num_elements)
            sparse_flat_indices = sparse_flat_indices.astype(int)
            sparse_indices = np.unravel_index(sparse_flat_indices, matrix_shape)
            sparse_indices = np.vstack(sparse_indices).transpose()
            values = np.random.randn(num_elements).astype(np.float32)
            sparse_2 = tf.SparseTensor(indices=sparse_indices, values=values,
                                       dense_shape=matrix_shape)
            res_actual = ops.flat_inner(tt_1, sparse_2)
            res_actual_val, tt_1_val = sess.run([res_actual, ops.full(tt_1)])
            res_desired_val = tt_1_val.flatten()[sparse_flat_indices].dot(values)
            self.assertAllClose(res_actual_val, res_desired_val)

  def testFrobeniusNormMatrix(self):
    # Frobenius norm of a TT-matrix.
    shape_list = (((2, 2), (3, 4)),
                  ((2, 3, 4), (2, 2, 2)))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for tensor_shape in shape_list:
        for rank in rank_list:
          tt = initializers.random_matrix(tensor_shape, tt_rank=rank)
          norm_sq_actual = ops.frobenius_norm_squared(tt)
          norm_actual = ops.frobenius_norm(tt)
          vars = [norm_sq_actual, norm_actual, ops.full(tt)]
          norm_sq_actual_val, norm_actual_val, tt_val = sess.run(vars)
          tt_val = tt_val.flatten()
          norm_sq_desired_val = tt_val.dot(tt_val)
          norm_desired_val = np.linalg.norm(tt_val)
          self.assertAllClose(norm_sq_actual_val, norm_sq_desired_val)
          self.assertAllClose(norm_actual_val, norm_desired_val, atol=1e-5,
                              rtol=1e-5)

  def testTranspose(self):
    # Transpose a TT-matrix.
    shape_list = (((2, 2), (3, 4)),
                  ((2, 3, 4), (2, 2, 2)))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for tensor_shape in shape_list:
        for rank in rank_list:
          tt = initializers.random_matrix(tensor_shape, tt_rank=rank)
          res_actual = ops.full(ops.transpose(tt))
          res_actual_val, tt_val = sess.run([res_actual, ops.full(tt)])
          self.assertAllClose(tt_val.transpose(), res_actual_val)

  def testQuadraticForm(self):
    # Test quadratic form.
    shape_list = (((2, 2), (3, 4)),
                  ((2, 3, 4), (2, 2, 2)))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for tensor_shape in shape_list:
        for rank in rank_list:
          A = initializers.random_matrix(tensor_shape, tt_rank=rank)
          b = initializers.random_matrix((tensor_shape[0], None), tt_rank=rank)
          c = initializers.random_matrix((tensor_shape[1], None), tt_rank=rank)
          res_actual = ops.quadratic_form(A, b, c)
          vars = [res_actual, ops.full(A), ops.full(b), ops.full(c)]
          res_actual_val, A_val, b_val, c_val = sess.run(vars)
          res_desired = b_val.T.dot(A_val).dot(c_val)
          self.assertAllClose(res_actual_val, np.squeeze(res_desired))

  def testCastFloat(self):
    # Test cast function for float tt-matrices and vectors.
    
    tt_mat = initializers.random_matrix(((2, 3), (3, 2)), tt_rank=2)

    tt_vec = initializers.random_matrix(((2, 3), None), tt_rank=2)
    
    for dtype in [tf.float16, tf.float32, tf.float64]:
      self.assertEqual(ops.cast(tt_vec, dtype).dtype, dtype)
      self.assertEqual(ops.cast(tt_mat, dtype).dtype, dtype)

  def testCastIntFloat(self):
    # Tests cast function from int to float for matrices.
    np.random.seed(1)
    K_1 = np.random.randint(0, high=100, size=(1, 2, 2, 2))
    K_2 = np.random.randint(0, high=100, size=(2, 3, 3, 2))
    K_3 = np.random.randint(0, high=100, size=(2, 2, 2, 1))
    tt_int = TensorTrain([K_1, K_2, K_3], tt_ranks=[1, 2, 2, 1])
    
    for dtype in [tf.float16, tf.float32, tf.float64]:
      self.assertEqual(ops.cast(tt_int, dtype).dtype, dtype)

  def testUnknownRanksTTMatmul(self):
    # Tests tt_tt_matmul for matrices with unknown ranks
    K_1 = tf.placeholder(tf.float32, (1, 2, 2, None))
    K_2 = tf.placeholder(tf.float32, (None, 3, 3, 1))
    tt_mat = TensorTrain([K_1, K_2])
    res_actual = ops.full(ops.matmul(tt_mat, tt_mat))
    res_desired = tf.matmul(ops.full(tt_mat), ops.full(tt_mat))
    np.random.seed(1)
    K_1_val = np.random.rand(1, 2, 2, 2)
    K_2_val = np.random.rand(2, 3, 3, 1)
    with self.test_session() as sess:
      res_actual_val = sess.run(res_actual, {K_1: K_1_val, K_2: K_2_val})
      res_desired_val = sess.run(res_desired, {K_1: K_1_val, K_2: K_2_val})
      self.assertAllClose(res_desired_val, res_actual_val)


  def testHalfKnownRanksTTMatmul(self):
    # Tests tt_tt_matmul for the case  when one matrice has known ranks 
    # and the other one doesn't    
    np.random.seed(1)
    K_1 = tf.placeholder(tf.float32, (1, 2, 2, None))
    K_2 = tf.placeholder(tf.float32, (None, 3, 3, 1))
    tt_mat_known_ranks = TensorTrain([K_1, K_2], tt_ranks=[1, 3, 1])
    tt_mat = TensorTrain([K_1, K_2])
    res_actual = ops.full(ops.matmul(tt_mat_known_ranks, tt_mat))
    res_desired = tf.matmul(ops.full(tt_mat_known_ranks), ops.full(tt_mat))
    np.random.seed(1)
    K_1_val = np.random.rand(1, 2, 2, 3)
    K_2_val = np.random.rand(3, 3, 3, 1)
    with self.test_session() as sess:
      res_actual_val = sess.run(res_actual, {K_1: K_1_val, K_2: K_2_val})
      res_desired_val = sess.run(res_desired, {K_1: K_1_val, K_2: K_2_val})
      self.assertAllClose(res_desired_val, res_actual_val)


class TTTensorBatchTest(tf.test.TestCase):

  def testFullTensor2d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(3, 10, rank).astype(np.float32)
      b = np.random.rand(3, rank, 9).astype(np.float32)
      tt_cores = (a.reshape(3, 1, 10, rank), b.reshape(3, rank, 9, 1))
      desired = np.einsum('oib,obj->oij', a, b)
      with self.test_session():
        tf_tens = TensorTrainBatch(tt_cores)
        actual = ops.full(tf_tens)
        self.assertAllClose(desired, actual.eval())

  def testFullTensor3d(self):
    np.random.seed(1)
    for rank_1 in [1, 2]:
      a = np.random.rand(3, 10, rank_1).astype(np.float32)
      b = np.random.rand(3, rank_1, 9, 3).astype(np.float32)
      c = np.random.rand(3, 3, 8).astype(np.float32)
      tt_cores = (a.reshape(3, 1, 10, rank_1), b, c.reshape((3, 3, 8, 1)))
      # Basically do full by hand.
      desired = np.einsum('oia,oajb,obk->oijk', a, b, c)
      with self.test_session():
        tf_tens = TensorTrainBatch(tt_cores)
        actual = ops.full(tf_tens)
        self.assertAllClose(desired, actual.eval())

  def testFlatInnerTTTensbyTTTensSameBatchSize(self):
    # Inner product between two batch TT-tensors of the same batch_size.
    shape_list = ((2, 2),
                  (2, 3, 4))
    rank_list = (1, 2)
    with self.test_session() as sess:
      for shape in shape_list:
        for rank in rank_list:
          tt_1 = initializers.random_tensor_batch(shape, tt_rank=rank,
                                                  batch_size=2)
          tt_2 = initializers.random_tensor_batch(shape, tt_rank=rank,
                                                  batch_size=2)
          res_actual = ops.flat_inner(tt_1, tt_2)
          tt_1_full = tf.reshape(ops.full(tt_1), (2, 1, -1))
          tt_2_full = tf.reshape(ops.full(tt_2), (2, -1, 1))
          res_desired = tf.matmul(tt_1_full, tt_2_full)
          res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
          self.assertAllClose(res_actual_val, np.squeeze(res_desired_val))

  def testFlatInnerTTTensbyTTTensBroadcasting(self):
    # Inner product between two batch TT-tensors with broadcasting.
    tt_1 = initializers.random_tensor_batch((2, 3, 4), batch_size=1)
    tt_2 = initializers.random_tensor_batch((2, 3, 4), batch_size=3)
    res_actual_1 = ops.flat_inner(tt_1, tt_2)
    res_actual_2 = ops.flat_inner(tt_2, tt_1)
    res_desired = tf.einsum('ijk,oijk->o', ops.full(tt_1[0]), ops.full(tt_2))
    with self.test_session() as sess:
      res = sess.run([res_actual_1, res_actual_2, res_desired])
      res_actual_1_val, res_actual_2_val, res_desired_val = res
      self.assertAllClose(res_actual_1_val, res_desired_val)
      self.assertAllClose(res_actual_2_val, res_desired_val)

    tt_1 = initializers.random_tensor_batch((2, 3, 4), batch_size=2)
    with self.assertRaises(ValueError):
      # The batch_sizes are different.
      ops.flat_inner(tt_1, tt_2)

  def testAddSameBatchSize(self):
    # Sum two TT-tensors with the same batch size.
    tt_a = initializers.random_tensor_batch((2, 1, 4), tt_rank=2, batch_size=3)
    tt_b = initializers.random_tensor_batch((2, 1, 4), tt_rank=[1, 2, 4, 1],
                                            batch_size=3)
    with self.test_session() as sess:
      res_actual = ops.full(ops.add(tt_a, tt_b))
      res_actual2 = ops.full(tt_a + tt_b)
      res_desired = ops.full(tt_a) + ops.full(tt_b)
      to_run = [res_actual, res_actual2, res_desired]
      res_actual_val, res_actual2_val, res_desired_val = sess.run(to_run)
      self.assertAllClose(res_actual_val, res_desired_val)
      self.assertAllClose(res_actual2_val, res_desired_val)

  def testAddBroadcasting(self):
    # Sum two TT-tensors with broadcasting.
    tt_a = initializers.random_tensor_batch((2, 1, 4), tt_rank=2, batch_size=1)
    tt_b = initializers.random_tensor_batch((2, 1, 4), tt_rank=[1, 2, 4, 1],
                                            batch_size=3)
    with self.test_session() as sess:
      res_actual = ops.full(ops.add(tt_a, tt_b))
      res_actual2 = ops.full(tt_b + tt_a)
      res_desired = ops.full(tt_a) + ops.full(tt_b)
      to_run = [res_actual, res_actual2, res_desired]
      res_actual_val, res_actual2_val, res_desired_val = sess.run(to_run)
      self.assertAllClose(res_actual_val, res_desired_val)
      self.assertAllClose(res_actual2_val, res_desired_val)

  def testFrobeniusNormTens(self):
    # Frobenius norm of a batch of TT-tensors.
    with self.test_session() as sess:
      tt = initializers.random_tensor_batch((2, 1, 3), batch_size=3)
      norm_sq_actual = ops.frobenius_norm_squared(tt)
      norm_actual = ops.frobenius_norm(tt)
      vars = [norm_sq_actual, norm_actual, ops.full(tt)]
      norm_sq_actual_val, norm_actual_val, tt_val = sess.run(vars)
      tt_val = tt_val.reshape((3, -1))
      norm_sq_desired_val = np.sum(tt_val * tt_val, axis=1)
      norm_desired_val = np.sqrt(norm_sq_desired_val)
      self.assertAllClose(norm_sq_actual_val, norm_sq_desired_val)
      self.assertAllClose(norm_actual_val, norm_desired_val, atol=1e-5,
                          rtol=1e-5)


class TTMatrixTestBatch(tf.test.TestCase):

  def testFullMatrix2d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(3, 2, 3, rank).astype(np.float32)
      b = np.random.rand(3, rank, 4, 5).astype(np.float32)
      tt_cores = (a.reshape(3, 1, 2, 3, rank), b.reshape((3, rank, 4, 5, 1)))
      # Basically do full by hand.
      desired = np.einsum('oijb,obkl->oijkl', a, b)
      desired = desired.reshape((3, 2, 3, 4, 5))
      desired = desired.transpose((0, 1, 3, 2, 4))
      desired = desired.reshape((3, 2 * 4, 3 * 5))
      with self.test_session():
        tf_mat = TensorTrainBatch(tt_cores)
        actual = ops.full(tf_mat)
        self.assertAllClose(desired, actual.eval())

  def testFullMatrix3d(self):
    np.random.seed(1)
    for rank in [1, 2]:
      a = np.random.rand(3, 2, 3, rank).astype(np.float32)
      b = np.random.rand(3, rank, 4, 5, rank).astype(np.float32)
      c = np.random.rand(3, rank, 2, 2).astype(np.float32)
      tt_cores = (a.reshape(3, 1, 2, 3, rank), b.reshape(3, rank, 4, 5, rank),
                  c.reshape(3, rank, 2, 2, 1))
      # Basically do full by hand.
      desired = np.einsum('oija,oaklb,obpq->oijklpq', a, b, c)
      desired = desired.reshape((3, 2, 3, 4, 5, 2, 2))
      desired = desired.transpose((0, 1, 3, 5, 2, 4, 6))
      desired = desired.reshape((3, 2 * 4 * 2, 3 * 5 * 2))
      with self.test_session():
        tf_mat = TensorTrainBatch(tt_cores)
        actual = ops.full(tf_mat)
        self.assertAllClose(desired, actual.eval())

  def testTTMatTimesTTMatSameBatchSize(self):
    # Multiply a batch of TT-matrices by another batch of TT-matrices with the
    # same batch sizes.
    left_shape = (2, 3)
    sum_shape = (4, 3)
    right_shape = (4, 4)
    with self.test_session() as sess:
      tt_mat_1 = initializers.random_matrix_batch((left_shape, sum_shape),
                                                  tt_rank=3, batch_size=3)
      tt_mat_2 = initializers.random_matrix_batch((sum_shape, right_shape),
                                                  batch_size=3)
      res_actual = ops.matmul(tt_mat_1, tt_mat_2)
      res_actual = ops.full(res_actual)
      res_desired = tf.matmul(ops.full(tt_mat_1), ops.full(tt_mat_2))
      res_actual_val, res_desired_val = sess.run([res_actual, res_desired])
      # TODO: why so bad accuracy?
      self.assertAllClose(res_actual_val, res_desired_val, atol=1e-5, rtol=1e-5)

  def testTTMatTimesTTMatBroadcasting(self):
    # Multiply a batch of TT-matrices by another batch of TT-matrices with
    # broadcasting.
    left_shape = (2, 3)
    sum_shape = (4, 3)
    right_shape = (4, 4)
    with self.test_session() as sess:
      tt_mat_1 = initializers.random_matrix_batch((left_shape, sum_shape),
                                                  tt_rank=3, batch_size=3)
      tt_mat_2 = initializers.random_matrix_batch((sum_shape, right_shape))
      # TT-batch by one element TT-batch
      res_actual = ops.matmul(tt_mat_1, tt_mat_2)
      res_actual = ops.full(res_actual)
      # TT by TT-batch.
      res_actual2 = ops.matmul(ops.transpose(tt_mat_2[0]), ops.transpose(tt_mat_1))
      res_actual2 = ops.full(ops.transpose(res_actual2))
      res_desired = tf.einsum('oij,jk->oik', ops.full(tt_mat_1),
                              ops.full(tt_mat_2[0]))
      to_run = [res_actual, res_actual2, res_desired]
      res_actual_val, res_actual2_val, res_desired_val = sess.run(to_run)
      self.assertAllClose(res_actual_val, res_desired_val)
      self.assertAllClose(res_actual2_val, res_desired_val)

  def testTranspose(self):
    # Transpose a batch of TT-matrices.
    with self.test_session() as sess:
      tt = initializers.random_matrix_batch(((2, 3, 4), (2, 2, 2)), batch_size=2)
      res_actual = ops.full(ops.transpose(tt))
      res_actual_val, tt_val = sess.run([res_actual, ops.full(tt)])
      self.assertAllClose(tt_val.transpose((0, 2, 1)), res_actual_val)

  def testAddSameBatchSize(self):
    # Sum two TT-matrices with the same batch size.
    tt_a = initializers.random_matrix_batch(((2, 1, 4), None), tt_rank=2,
                                            batch_size=3)
    tt_b = initializers.random_matrix_batch(((2, 1, 4), None),
                                            tt_rank=[1, 2, 4, 1], batch_size=3)
    with self.test_session() as sess:
      res_actual = ops.full(ops.add(tt_a, tt_b))
      res_actual2 = ops.full(tt_a + tt_b)
      res_desired = ops.full(tt_a) + ops.full(tt_b)
      to_run = [res_actual, res_actual2, res_desired]
      res_actual_val, res_actual2_val, res_desired_val = sess.run(to_run)
      self.assertAllClose(res_actual_val, res_desired_val)
      self.assertAllClose(res_actual2_val, res_desired_val)

  def testAddBroadcasting(self):
    # Sum two TT-matrices with broadcasting.
    tt_a = initializers.random_matrix_batch(((2, 1, 4), (2, 2, 2)), tt_rank=2,
                                            batch_size=3)
    tt_b = initializers.random_matrix_batch(((2, 1, 4), (2, 2, 2)),
                                            tt_rank=[1, 2, 4, 1], batch_size=1)
    with self.test_session() as sess:
      res_actual = ops.full(ops.add(tt_a, tt_b))
      res_actual2 = ops.full(tt_b + tt_a)
      res_desired = ops.full(tt_a) + ops.full(tt_b)
      to_run = [res_actual, res_actual2, res_desired]
      res_actual_val, res_actual2_val, res_desired_val = sess.run(to_run)
      self.assertAllClose(res_actual_val, res_desired_val)
      self.assertAllClose(res_actual2_val, res_desired_val)


def _random_sparse(shape, non_zeros):
  sparse_flat_indices = np.random.choice(np.prod(shape), non_zeros).astype(int)
  sparse_indices = np.unravel_index(sparse_flat_indices, shape)
  sparse_indices = np.vstack(sparse_indices).transpose()
  values = np.random.randn(non_zeros).astype(np.float32)
  sparse = tf.SparseTensor(indices=sparse_indices, values=values,
                             dense_shape=shape)
  return sparse

if __name__ == "__main__":
  tf.test.main()
