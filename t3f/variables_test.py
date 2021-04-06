import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from t3f import variables
from t3f import ops
from t3f import initializers


class _VariablesTest():

  def testAttributes(self):
    # Test that after converting an initializer into a variable all the
    # attributes stays the same.
    tens = initializers.random_tensor([2, 3, 2], tt_rank=2, dtype=self.dtype)
    tens_v = variables.get_variable('tt_tens', initializer=tens)
    mat = initializers.random_matrix([[3, 2, 2], [3, 3, 3]], tt_rank=3,
                                     dtype=self.dtype)
    mat_v = variables.get_variable('tt_mat', initializer=mat)
    for (init, var) in [[tens, tens_v], [mat, mat_v]]:
      self.assertEqual(init.get_shape(), var.get_shape())
      self.assertEqual(init.get_raw_shape(), var.get_raw_shape())
      self.assertEqual(init.ndims(), var.ndims())
      self.assertEqual(init.get_tt_ranks(), var.get_tt_ranks())
      self.assertEqual(init.is_tt_matrix(), var.is_tt_matrix())

  def testAssign(self):
    old_init = initializers.random_tensor([2, 3, 2], tt_rank=2,
                                          dtype=self.dtype)
    tt = variables.get_variable('tt', initializer=old_init)
    new_init = initializers.random_tensor([2, 3, 2], tt_rank=2,
                                          dtype=self.dtype)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    init_value =  self.evaluate(ops.full(tt))
    assigner = variables.assign(tt, new_init)
    assigner_value = self.evaluate(ops.full(assigner))
    after_value = ops.full(tt)
    after_value = self.evaluate(after_value)
    self.assertAllClose(assigner_value, after_value)
    # Assert that the value actually changed:
    abs_diff = np.linalg.norm((init_value - after_value).flatten())
    rel_diff = abs_diff / np.linalg.norm((init_value).flatten())
    self.assertGreater(rel_diff, 0.2)


class VariablesTestFloat32(tf.test.TestCase, _VariablesTest):
  dtype = tf.float32


class VariablesTestFloat64(tf.test.TestCase, _VariablesTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
