import numpy as np
import tensorflow as tf

import t3f

class TTArrayTest(tf.test.TestCase):

    def testTTVector(self):
        vec_shape = (2, 3, 4, 3)
        np.random.seed(1)
        vec = np.random.rand(*vec_shape).astype(np.float32)
        tt_vec = t3f.convert_to_tt_array(vec)
        self.assertAllClose(t3f.full(tt_vec), vec)


if __name__ == "__main__":
    tf.test.main()
