import numpy as np
import tensorflow as tf

import variables
import ops
import initializers

class VariablesTest(tf.test.TestCase):

  def testGetExistingVariable(self):
    init = initializers.tt_rand_tensor([2, 3, 2], tt_rank=2)
    tt_1 = variables.get_tt_variable('tt_1', initializer=init)
    # Check that we can create another variable without name
    # conflict (ValueError).
    variables.get_tt_variable('tt_2', initializer=init)
    with self.test_session():
      tf.global_variables_initializer().run()
      with self.assertRaises(ValueError):
        # The variable already exists and scope.reuse is False by default.
        variables.get_tt_variable('tt_1')
      with tf.variable_scope('', reuse=True):
        tt_1_copy = variables.get_tt_variable('tt_1')
        self.assertAllClose(ops.full(tt_1).eval(), ops.full(tt_1_copy).eval())

  def testAssign(self):
    old_init = initializers.tt_rand_tensor([2, 3, 2], tt_rank=2)
    tt = variables.get_tt_variable('tt', initializer=old_init)
    # The ranks of old_init and new_init should be different to check that
    # assign correctly changes the shape of the TT-cores.
    new_init = initializers.tt_rand_tensor([2, 3, 2], tt_rank=4)
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
