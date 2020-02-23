import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from t3f.tensor_train import TensorTrain
from t3f.tensor_train_batch import TensorTrainBatch
from t3f import ops
from t3f import shapes
from t3f import initializers


class _TTMatrixTest():

  def testUnknownRanksTTMatmul(self):
    # Tests tt_tt_matmul for matrices with unknown ranks
    K_1 = tf.placeholder(self.dtype, (1, 2, 2, None))
    K_2 = tf.placeholder(self.dtype, (None, 3, 3, 1))
    tt_mat = TensorTrain([K_1, K_2])
    res_actual = ops.full(ops.matmul(tt_mat, tt_mat))
    res_desired = tf.matmul(ops.full(tt_mat), ops.full(tt_mat))
    np.random.seed(1)
    K_1_val = np.random.rand(1, 2, 2, 2)
    K_2_val = np.random.rand(2, 3, 3, 1)
    with tf.Session() as sess:
      res_actual_val = sess.run(res_actual, {K_1: K_1_val, K_2: K_2_val})
      res_desired_val = sess.run(res_desired, {K_1: K_1_val, K_2: K_2_val})
    self.assertAllClose(res_desired_val, res_actual_val)

  def testHalfKnownRanksTTMatmul(self):
    # Tests tt_tt_matmul for the case  when one matrice has known ranks
    # and the other one doesn't
    np.random.seed(1)
    K_1 = tf.placeholder(self.dtype, (1, 2, 2, None))
    K_2 = tf.placeholder(self.dtype, (None, 3, 3, 1))
    tt_mat_known_ranks = TensorTrain([K_1, K_2], tt_ranks=[1, 3, 1])
    tt_mat = TensorTrain([K_1, K_2])
    res_actual = ops.full(ops.matmul(tt_mat_known_ranks, tt_mat))
    res_desired = tf.matmul(ops.full(tt_mat_known_ranks), ops.full(tt_mat))
    np.random.seed(1)
    K_1_val = np.random.rand(1, 2, 2, 3)
    K_2_val = np.random.rand(3, 3, 3, 1)
    with tf.Session() as sess:
      res_actual_val = sess.run(res_actual, {K_1: K_1_val, K_2: K_2_val})
      res_desired_val = sess.run(res_desired, {K_1: K_1_val, K_2: K_2_val})
    self.assertAllClose(res_desired_val, res_actual_val)


class _TTTensorBatchTest():

  def testMultiplyUnknownBatchSizeBroadcasting(self):
    c1 = tf.placeholder(self.dtype, [None, 1, 3, 2])
    c2 = tf.placeholder(self.dtype, [None, 2, 3, 1])
    tt_a = TensorTrainBatch([c1, c2])
    tt_b = initializers.random_tensor_batch((3, 3), tt_rank=3, batch_size=1,
                                            dtype=self.dtype)
    tt_c = initializers.random_tensor((3, 3), tt_rank=3,
                                      dtype=self.dtype)
    res_ab = ops.full(ops.multiply(tt_a, tt_b))
    res_ba = ops.full(ops.multiply(tt_b, tt_a))
    res_ac = ops.full(ops.multiply(tt_a, tt_c))
    res_ca = ops.full(ops.multiply(tt_c, tt_a))
    res_desired_ab = ops.full(tt_a) * ops.full(tt_b)
    res_desired_ac = ops.full(tt_a) * ops.full(tt_c)
    to_run = [res_ab, res_ba, res_ac, res_ca, res_desired_ab, res_desired_ac]
    feed_dict = {c1:np.random.rand(7, 1, 3, 2),
                 c2:np.random.rand(7, 2, 3, 1)}
    with tf.Session() as sess:
      ab, ba, ac, ca, des_ab, des_ac = sess.run(to_run, feed_dict=feed_dict)
    self.assertAllClose(ab, des_ab)
    self.assertAllClose(ba, des_ab)
    self.assertAllClose(ac, des_ac)
    self.assertAllClose(ca, des_ac)

  def testMultiplyTwoBatchesUnknownSize(self):
    c1 = tf.placeholder(self.dtype, [None, 1, 3, 2])
    c2 = tf.placeholder(self.dtype, [None, 2, 3, 1])
    c3 = tf.placeholder(self.dtype, [None, 1, 3, 2])
    c4 = tf.placeholder(self.dtype, [None, 2, 3, 1])
    tt_a = TensorTrainBatch([c1, c2])
    tt_b = TensorTrainBatch([c3, c4])
    res_ab = ops.full(ops.multiply(tt_a, tt_b))
    res_ba = ops.full(ops.multiply(tt_b, tt_a))
    res_desired = ops.full(tt_a) * ops.full(tt_b)
    to_run = [res_ab, res_ba, res_desired]
    feed_dict = {c1:np.random.rand(7, 1, 3, 2),
                 c2:np.random.rand(7, 2, 3, 1),
                 c3:np.random.rand(7, 1, 3, 2),
                 c4:np.random.rand(7, 2, 3, 1)}

    feed_dict_err = {c1:np.random.rand(7, 1, 3, 2),
                     c2:np.random.rand(7, 2, 3, 1),
                     c3:np.random.rand(1, 1, 3, 2),
                     c4:np.random.rand(1, 2, 3, 1)}

    with tf.Session() as sess:
      ab_full, ba_full, des_full = sess.run(to_run, feed_dict=feed_dict)
      self.assertAllClose(ab_full, des_full)
      self.assertAllClose(ba_full, des_full)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(to_run, feed_dict=feed_dict_err)

  def testMultiplyUnknownSizeBatchAndBatch(self):
    c1 = tf.placeholder(self.dtype, [None, 1, 3, 2])
    c2 = tf.placeholder(self.dtype, [None, 2, 3, 1])
    tt_b = initializers.random_tensor_batch((3, 3), tt_rank=2, batch_size=8,
                                            dtype=self.dtype)
    tt_a = TensorTrainBatch([c1, c2])
    res_ab = ops.full(ops.multiply(tt_a, tt_b))
    res_ba = ops.full(ops.multiply(tt_b, tt_a))
    res_desired = ops.full(tt_a) * ops.full(tt_b)
    to_run = [res_ab, res_ba, res_desired]
    feed_dict = {c1:np.random.rand(8, 1, 3, 2),
                 c2:np.random.rand(8, 2, 3, 1)}

    feed_dict_err = {c1:np.random.rand(1, 1, 3, 2),
                     c2:np.random.rand(1, 2, 3, 1)}

    with tf.Session() as sess:
      ab_full, ba_full, des_full = sess.run(to_run, feed_dict=feed_dict)
      self.assertAllClose(ab_full, des_full)
      self.assertAllClose(ba_full, des_full)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(to_run, feed_dict=feed_dict_err)

  def testGatherND(self):
    idx = [[0, 0, 0], [0, 1, 2], [0, 1, 0]]
    pl_idx = tf.placeholder(tf.int32, [None, 3])
    tt = initializers.random_tensor((3, 4, 5), tt_rank=2, dtype=self.dtype)
    res_np = ops.gather_nd(tt, idx)
    res_pl = ops.gather_nd(tt, pl_idx)
    res_desired = tf.gather_nd(ops.full(tt), idx)
    to_run = [res_np, res_pl, res_desired]
    with tf.Session() as sess:
      res_np_v, res_pl_v, des_v = sess.run(to_run, feed_dict={pl_idx: idx})
    self.assertAllClose(res_np_v, des_v)
    self.assertAllClose(res_pl_v, res_pl_v)

  def testGatherNDBatch(self):
    idx = [[0, 0, 0, 0], [1, 0, 1, 2], [0, 0, 1, 0]]
    pl_idx = tf.placeholder(tf.int32, [None, 4])
    tt = initializers.random_tensor_batch((3, 4, 5), tt_rank=2, batch_size=2,
                                          dtype=self.dtype)
    res_np = ops.gather_nd(tt, idx)
    res_pl = ops.gather_nd(tt, pl_idx)
    res_desired = tf.gather_nd(ops.full(tt), idx)
    to_run = [res_np, res_pl, res_desired]
    with tf.Session() as sess:
      res_np_v, res_pl_v, des_v = sess.run(to_run, feed_dict={pl_idx: idx})
    self.assertAllClose(res_np_v, des_v)
    self.assertAllClose(res_pl_v, res_pl_v)


class TTMatrixTestFloat32(tf.test.TestCase, _TTMatrixTest):
  dtype = tf.float32


class TTMatrixTestFloat64(tf.test.TestCase, _TTMatrixTest):
  dtype = tf.float64


class TTTensorBatchTestFloat32(tf.test.TestCase, _TTTensorBatchTest):
  dtype = tf.float32


class TTTensorBatchTestFloat64(tf.test.TestCase, _TTTensorBatchTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
