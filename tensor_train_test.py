import numpy as np
import tensorflow as tf

import tensor_train
import initializers
import ops


class TensorTrainTest(tf.test.TestCase):

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
                if desired:
                    tf_tens = tensor_train.TensorTrain((a, b))
                    tf_cores = tf_tens.tt_cores
                    self.assertEqual(desired, tensor_train._are_tt_cores_valid(tf_cores))
                else:
                    with self.assertRaises(ValueError):
                        tensor_train.TensorTrain((a, b))

                # Make dtypes inconsistent.
                b_new = b.astype(np.float64)
                self.assertEqual(False, tensor_train._are_tt_cores_valid((a, b_new)))
                with self.assertRaises(ValueError):
                    tensor_train.TensorTrain((a, b_new))

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
                if desired:
                    tf_tens = tensor_train.TensorTrain((a, b, c))
                    tf_cores = tf_tens.tt_cores
                    self.assertEqual(desired, tensor_train._are_tt_cores_valid(tf_cores))
                else:
                    with self.assertRaises(ValueError):
                        tensor_train.TensorTrain((a, b, c))

                # Make dtypes inconsistent.
                b_new = b.astype(np.float64)
                self.assertEqual(False, tensor_train._are_tt_cores_valid((a, b_new, c)))
                with self.assertRaises(ValueError):
                    tensor_train.TensorTrain((a, b_new, c))

    def testTensorIndexing(self):
        # TODO: random seed.
        tens = initializers.tt_rand_tensor((3, 3, 4))
        with self.test_session() as sess:
            desired = ops.full(tens)[:, :, :]
            actual = ops.full(tens[:, :, :])
            desired, actual = sess.run([desired, actual])
            self.assertAllClose(desired, actual)
            desired = ops.full(tens)[1, :, :]
            actual = ops.full(tens[1, :, :])
            desired, actual = sess.run([desired, actual])
            self.assertAllClose(desired, actual)
            desired = ops.full(tens)[1:2, 1, :]
            actual = ops.full(tens[1:2, 1, :])
            desired, actual = sess.run([desired, actual])
            self.assertAllClose(desired, actual)
            desired = ops.full(tens)[0:3, :, 3]
            actual = ops.full(tens[0:3, :, 3])
            desired, actual = sess.run([desired, actual])
            self.assertAllClose(desired, actual)
            desired = ops.full(tens)[1, 2, 3]
            actual = ops.full(tens[1, 2, 3])
            desired, actual = sess.run([desired, actual])
            self.assertAllClose(desired, actual)
            desired = ops.full(tens)[1, :, 3]
            actual = ops.full(tens[1, :, 3])
            desired, actual = sess.run([desired, actual])
            self.assertAllClose(desired, actual)



if __name__ == "__main__":
    tf.test.main()
