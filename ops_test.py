import numpy as np
import tensorflow as tf

import tensor_train
import ops


class TTTensorTest(tf.test.TestCase):

    def testValidateTTCores2d(self):
        np.random.seed(1)
        schedule = (((1, 1, 1, 1), True),
                    ((1, 2, 2, 1), True),
                    ((2, 1, 1, 1), False),
                    ((1, 1, 1, 2), False),
                    ((1, 1, 2, 1), False),
                    ((1, 2, 1, 1), False))

        for ranks, desired in schedule:
            a = np.random.rand(ranks[0], 10, ranks[1]).astype(np.float32)
            b = np.random.rand(ranks[2], 9, ranks[3]).astype(np.float32)
            with self.test_session():
                self.assertEqual(desired, tensor_train._are_tt_cores_valid((a, b)))
                tf_tens = tensor_train.TensorTrain((a, b))
                tf_cores = tf_tens.tt_cores()
                self.assertEqual(desired, tensor_train._are_tt_cores_valid(tf_cores))

                b = b.astype(np.float64)
                self.assertEqual(desired, tensor_train._are_tt_cores_valid((a, b)))
                tf_tens = tensor_train.TensorTrain((a, b))
                tf_cores = tf_tens.tt_cores()
                self.assertEqual(desired, tensor_train._are_tt_cores_valid(tf_cores))

        def testValidateTTCores3d(self):
            np.random.seed(1)
            schedule = (((1, 1, 1, 1, 1, 1), True),
                        ((1, 2, 2, 2, 2, 1), True),
                        ((2, 1, 1, 1, 1, 1), False),
                        ((1, 1, 1, 1, 1, 2), False),
                        ((1, 1, 2, 1, 1, 1), False),
                        ((1, 2, 1, 1, 1, 1), False),
                        ((1, 2, 2, 1, 1, 1), True),
                        ((1, 2, 2, 3, 3, 1), True))

            for ranks, desired in schedule:
                a = np.random.rand(ranks[0], 10, ranks[1]).astype(np.float32)
                b = np.random.rand(ranks[2], 1, ranks[3]).astype(np.float32)
                c = np.random.rand(ranks[4], 2, ranks[5]).astype(np.float32)
                with self.test_session():
                    self.assertEqual(desired, tensor_train._are_tt_cores_valid((a, b, c)))
                    tf_tens = tensor_train.TensorTrain((a, b, c))
                    tf_cores = tf_tens.tt_cores()
                    self.assertEqual(desired, tensor_train._are_tt_cores_valid(tf_cores))

                    b = b.astype(np.float64)
                    self.assertEqual(False, tensor_train._are_tt_cores_valid((a, b, c)))
                    tf_tens = tensor_train.TensorTrain((a, b, c))
                    tf_cores = tf_tens.tt_cores()
                    self.assertEqual(False, tensor_train._are_tt_cores_valid(tf_cores))

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
            tt_mat = ops.to_tt_matrix(tf_mat, ((out_shape, inp_shape)))
            self.assertAllClose(mat, ops.full(tt_mat).eval())

    def testTTMatTimesDenseVec(self):
        # Multiply a TT-matrix by a dense vector.
        inp_shape = (2, 3, 4)
        out_shape = (3, 4, 3)
        np.random.seed(1)
        vec = np.random.rand(np.prod(inp_shape), 1).astype(np.float32)
        with self.test_session():
            tf_vec = tf.constant(vec)
            tf.set_random_seed(1)
            tt_mat = ops.tt_rand_matrix((out_shape, inp_shape))
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
            tt_vec = ops.tt_rand_matrix((inp_shape, None))
            res_actual = ops.matmul(tf_mat, tt_vec)
            res_desired = tf.matmul(ops.full(tf_mat), tt_vec)
            self.assertAllClose(res_actual.eval(), res_desired.eval())


if __name__ == "__main__":
    tf.test.main()
