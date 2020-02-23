# Graph mode tests.
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from t3f import ops
from t3f import shapes
from t3f import decompositions


class _DecompositionsTest():

  def testTTTensor(self):
    # Test that a tensor of ones and of zeros can be converted into TT with
    # TT-rank 1.
    shape = (2, 1, 4, 3)
    tens_arr = (np.zeros(shape).astype(self.dtype.as_numpy_dtype),
                np.ones(shape).astype(self.dtype.as_numpy_dtype))
    for tens in tens_arr:
      with tf.Session() as sess:
        tf_tens = tf.constant(tens)
        tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=1)
        static_tt_ranks = tt_tens.get_tt_ranks().as_list()

        # Try to decompose the same tensor with unknown shape.
        tf_tens_pl = tf.placeholder(self.dtype, (None, None, None, None))
        tt_tens = decompositions.to_tt_tensor(tf_tens_pl, max_tt_rank=1)
        tt_val = ops.full(tt_tens).eval({tf_tens_pl: tens})
        self.assertAllClose(tens, tt_val)
        dynamic_tt_ranks = shapes.tt_ranks(tt_tens).eval({tf_tens_pl: tens})
        self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)


class DecompositionsTestFloat32(tf.test.TestCase, _DecompositionsTest):
  dtype = tf.float32


class DecompositionsTestFloat64(tf.test.TestCase, _DecompositionsTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
