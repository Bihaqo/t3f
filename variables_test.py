import tensorflow as tf

import variables
import ops


class VariablesTest(tf.test.TestCase):

  def testGetExistingVariable(self):
    tt_1 = variables.get_tt_variable('tt_1', shape=[2, 3, 2], rank=2)
    # Check that we can create new variable without name conflict (ValueError).
    tt_2 = variables.get_tt_variable('tt_2', shape=[2, 3, 2], rank=2)
    with self.test_session():
      tf.global_variables_initializer().run()
      with tf.variable_scope('', reuse=True):
        tt_1_copy = variables.get_tt_variable('tt_1')
        self.assertAllClose(ops.full(tt_1).eval(), ops.full(tt_1_copy).eval())
      with self.assertRaises(ValueError):
        variables.get_tt_variable('tt_1')


if __name__ == "__main__":
  tf.test.main()
