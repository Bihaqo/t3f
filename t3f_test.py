import numpy as np
import tensorflow as tf

import t3f

class TTMatrixTest(tf.test.TestCase):

    def testTTVector(self):
        vec_shape = (2, 3, 4, 3)
        np.random.seed(1)
        vec = np.random.rand(np.prod(vec_shape)).astype(np.float32)
        with self.test_session():
            tf_vec = tf.constant(vec)
            tt_vec = t3f.to_tt_matrix(tf_vec, vec_shape)
            # TODO: test full and to_tt_matrix separately?
            self.assertAllClose(vec, t3f.full_matrix(tt_vec).eval())


    def testTTMatrix(self):
        # Convert a np.prod(out_shape) x np.prod(in_shape) matrix into TT-matrix
        # and back.
        inp_shape = (2, 3, 4, 3)
        out_shape = (3, 3, 3, 3)
        np.random.seed(1)
        mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape)).astype(np.float32)
        with self.test_session():
            tf_mat = tf.constant(mat)
            tt_mat = t3f.to_tt_matrix(tf_mat, ((out_shape, inp_shape)))
            # TODO: test full and to_tt_matrix separately?
            self.assertAllClose(mat, t3f.full_matrix(tt_mat).eval())
        tt_mat = t3f.to_tt_matrix(mat, ((out_shape, inp_shape)))

    def testTTmatTimesDenseVec(self):
        # Multiply a TT-matrix by a dense vector.
        inp_shape = (2, 3, 4, 3)
        out_shape = (3, 3, 3, 3)
        np.random.seed(1)
        vec = np.random.rand(np.prod(inp_shape), 1).astype(np.float32)
        with self.test_session():
            tf_vec = tf.constant(vec)
            tf.set_random_seed(1)
            tt_mat = t3f.tt_rand_matrix(((out_shape, inp_shape)))
            res_actual = t3f.matmul(tt_mat, tf_vec)
            res_desired = tf.matmul(t3f.full_matrix(tt_mat), tf_vec)
            self.assertAllClose(res_actual.eval(), res_desired.eval())

if __name__ == "__main__":
    tf.test.main()
