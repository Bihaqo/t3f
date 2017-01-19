import numpy as np
import tensorflow as tf

import t3f

class TTMatrixTest(tf.test.TestCase):

    def testTTVector(self):
        vec_shape = (2, 3, 4, 3)
        np.random.seed(1)
        vec = np.random.rand(*vec_shape).astype(np.float32)
        tt_vec = t3f.to_tt_matrix(vec, vec_shape)
        # TODO: test full and to_tt_array separately?
        self.assertAllClose(t3f.full_matrix(tt_vec), vec)

        vec = np.random.rand(np.prod(vec_shape)).astype(np.float32)
        tt_vec = t3f.to_tt_matrix(vec, vec_shape)
        # TODO: test full and to_tt_array separately?
        self.assertAllClose(t3f.full_matrix(tt_vec), vec)


    def testTTMatrix(self):
        # Convert a np.prod(out_shape) x np.prod(in_shape) matrix into TT-matrix
        # and back.
        inp_shape = (2, 3, 4, 3)
        out_shape = (3, 3, 3, 3)
        np.random.seed(1)
        mat = np.random.rand(np.prod(out_shape), np.prod(inp_shape)).astype(np.float32)
        tt_mat = t3f.to_tt_matrix(mat, ((out_shape, inp_shape)))
        # TODO: test full and to_tt_array separately?
        self.assertAllClose(t3f.full_matrix(tt_mat), mat)


if __name__ == "__main__":
    tf.test.main()
