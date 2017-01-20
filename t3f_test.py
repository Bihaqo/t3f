import numpy as np
import tensorflow as tf

import t3f

class TTMatrixTest(tf.test.TestCase):

    def testFullTensor2d(self):
        np.random.seed(1)
        for rank in [1, 2]:
            a = np.random.rand(10, rank).astype(np.float32)
            b = np.random.rand(rank, 9).astype(np.float32)
            tt_cores = (a.reshape(1, 10, rank), b.reshape(rank, 9, 1))
            desired = np.dot(a, b)
            with self.test_session():
                tf_tens = t3f.to_tensor_from_np(tt_cores)
                actual = t3f.full_tensor(tf_tens)
                self.assertAllClose(desired, actual.eval())

    def testFullTensor3d(self):
        np.random.seed(1)
        for rank_1 in [1, 2]:
            a = np.random.rand(10, rank_1).astype(np.float32)
            b = np.random.rand(rank_1, 9, 3).astype(np.float32)
            c = np.random.rand(3, 8).astype(np.float32)
            tt_cores = (a.reshape(1, 10, rank_1), b, c.reshape((3, 8, 1)))
            # Basically do full_tensor by hand.
            desired = a.dot(b.reshape((rank_1, -1)))
            desired = desired.reshape((-1, 3)).dot(c)
            desired = desired.reshape(10, 9, 8)
            with self.test_session():
                tf_tens = t3f.to_tensor_from_np(tt_cores)
                actual = t3f.full_tensor(tf_tens)
                self.assertAllClose(desired, actual.eval())

    def testFullMatrix2d(self):
        np.random.seed(1)
        for rank in [1, 2]:
            a = np.random.rand(2, 3, rank).astype(np.float32)
            b = np.random.rand(rank, 4, 5).astype(np.float32)
            tt_cores = (a.reshape(1, 2, 3, rank), b.reshape((rank, 4, 5, 1)))
            # Basically do full_matrix by hand.
            desired = a.reshape((-1, rank)).dot(b.reshape((rank, -1)))
            desired = desired.reshape((2, 3, 4, 5))
            desired = desired.transpose((0, 2, 1, 3))
            with self.test_session():
                tf_mat = t3f.to_tensor_from_np(tt_cores)
                actual = t3f.full_matrix(tf_mat)
                self.assertAllClose(desired, actual.eval())

    def testTTVector(self):
        vec_shape = (2, 1, 4, 3)
        np.random.seed(1)
        vec = np.random.rand(np.prod(vec_shape)).astype(np.float32)
        with self.test_session():
            tf_vec = tf.constant(vec)
            tt_vec = t3f.to_tt_matrix(tf_vec, vec_shape)
            self.assertAllClose(vec, t3f.full_matrix(tt_vec).eval())

    def testTTMatrix(self):
        # Convert a np.prod(out_shape) x np.prod(in_shape) matrix into TT-matrix
        # and back.
        inp_shape = (2, 5, 2, 3)
        out_shape = (3, 3, 2, 3)
        np.random.seed(1)
        mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape)).astype(np.float32)
        with self.test_session():
            tf_mat = tf.constant(mat)
            tt_mat = t3f.to_tt_matrix(tf_mat, ((out_shape, inp_shape)))
            self.assertAllClose(mat, t3f.full_matrix(tt_mat).eval())
        tt_mat = t3f.to_tt_matrix(mat, ((out_shape, inp_shape)))

    def testTTMatTimesDenseVec(self):
        # Multiply a TT-matrix by a dense vector.
        inp_shape = (2, 3, 4)
        out_shape = (3, 4, 3)
        np.random.seed(1)
        vec = np.random.rand(np.prod(inp_shape), 1).astype(np.float32)
        with self.test_session():
            tf_vec = tf.constant(vec)
            tf.set_random_seed(1)
            tt_mat = t3f.tt_rand_matrix((out_shape, inp_shape))
            res_actual = t3f.matmul(tt_mat, tf_vec)
            res_desired = tf.matmul(t3f.full_matrix(tt_mat), tf_vec)
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
            tt_vec = t3f.tt_rand_matrix(inp_shape)
            res_actual = t3f.matmul(tf_mat, tt_vec)
            res_desired = tf.matmul(t3f.full_matrix(tf_mat), tt_vec)
            self.assertAllClose(res_actual.eval(), res_desired.eval())


class TTTensorTest(tf.test.TestCase):

    def testTTTensor(self):
        shape = (2, 1, 4, 3)
        np.random.seed(1)
        tens = np.random.rand(*shape).astype(np.float32)
        with self.test_session():
            tf_tens = tf.constant(tens)
            tt_tens = t3f.to_tt_matrix(tf_tens, shape)
            self.assertAllClose(tens, t3f.full_matrix(tt_tens).eval())

if __name__ == "__main__":
    tf.test.main()
