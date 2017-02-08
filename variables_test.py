import numpy as np
import tensorflow as tf

import variables
import ops
import initializers

class VariablesTest(tf.test.TestCase):

  def testGetExistingVariable(self):
    init = initializers.random_tensor([2, 3, 2], tt_rank=2)
    tt_1 = variables.get_variable('tt_1', initializer=init)
    with tf.variable_scope('test'):
      tt_2 = variables.get_variable('tt_2', initializer=init)
    with self.test_session():
      tf.global_variables_initializer().run()
      with self.assertRaises(ValueError):
        # The variable already exists and scope.reuse is False by default.
        variables.get_variable('tt_1')
      with self.assertRaises(ValueError):
        with tf.variable_scope('', reuse=True):
          # The variable doesn't exist.
          variables.get_variable('tt_3')
      with tf.variable_scope('', reuse=True):
        tt_1_copy = variables.get_variable('tt_1')
        self.assertAllClose(ops.full(tt_1).eval(), ops.full(tt_1_copy).eval())

      with self.assertRaises(ValueError):
        with tf.variable_scope('', reuse=True):
          # The variable is defined in a different scope
          variables.get_variable('tt_2')

      with self.assertRaises(ValueError):
        with tf.variable_scope('nottest', reuse=True):
          # The variable is defined in a different scope
          variables.get_variable('tt_2')

      with tf.variable_scope('test', reuse=True):
        tt_2_copy = variables.get_variable('tt_2')
        self.assertAllClose(ops.full(tt_2).eval(), ops.full(tt_2_copy).eval())

  def testAttributes(self):
    # Test that after converting an initializer into a variable all the
    # attributes stays the same.
    tens = initializers.random_tensor([2, 3, 2], tt_rank=2)
    tens_v = variables.get_variable('tt_tens', initializer=tens)
    mat = initializers.random_matrix([[3, 2, 2], [3, 3, 3]], tt_rank=3)
    mat_v = variables.get_variable('tt_mat', initializer=mat)
    for (init, var) in [[tens, tens_v], [mat, mat_v]]:
      self.assertEqual(init.get_shape(), var.get_shape())
      self.assertEqual(init.get_raw_shape(), var.get_raw_shape())
      self.assertEqual(init.ndims(), var.ndims())
      self.assertEqual(init.get_tt_ranks(), var.get_tt_ranks())
      self.assertEqual(init.is_tt_matrix(), var.is_tt_matrix())

  def testAssign(self):
    old_init = initializers.random_tensor([2, 3, 2], tt_rank=2)
    tt = variables.get_variable('tt', initializer=old_init)
    new_init = initializers.random_tensor([2, 3, 2], tt_rank=2)
    assigner = variables.assign(tt, new_init)
    with self.test_session():
      tf.global_variables_initializer().run()
      init_value = ops.full(tt).eval()
      assigner_value = ops.full(assigner).eval()
      after_value = ops.full(tt)
      after_value = after_value.eval()
      self.assertAllClose(assigner_value, after_value)
      # Assert that the value actually changed:
      abs_diff = np.linalg.norm((init_value - after_value).flatten())
      rel_diff = abs_diff / np.linalg.norm((init_value).flatten())
      self.assertGreater(rel_diff, 0.2)


if __name__ == "__main__":
  tf.test.main()
