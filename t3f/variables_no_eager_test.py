import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from t3f import variables
from t3f import ops
from t3f import initializers


class _VariablesTest():

  def testGetExistingVariable(self):
    init = initializers.random_tensor([2, 3, 2], tt_rank=2, dtype=self.dtype)
    tt_1 = variables.get_variable('tt_1', initializer=init)
    with tf.variable_scope('test'):
      tt_2 = variables.get_variable('tt_2', initializer=init)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      with self.assertRaises(ValueError):
        # The variable already exists and scope.reuse is False by default.
        variables.get_variable('tt_1')
      with self.assertRaises(ValueError):
        with tf.variable_scope('', reuse=True):
          # The variable doesn't exist.
          variables.get_variable('tt_3')

      with tf.variable_scope('', reuse=True):
        tt_1_copy = variables.get_variable('tt_1', dtype=self.dtype)
        self.assertAllClose(ops.full(tt_1).eval(), ops.full(tt_1_copy).eval())

      with tf.variable_scope('', reuse=True):
        # Again try to retrieve an existing variable, but pass an initializer
        # and check that it still works.
        tt_1_copy = variables.get_variable('tt_1', initializer=0 * init,
                                           dtype=self.dtype)
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
        tt_2_copy = variables.get_variable('tt_2', dtype=self.dtype)
        self.assertAllClose(ops.full(tt_2).eval(), ops.full(tt_2_copy).eval())


class VariablesTestFloat32(tf.test.TestCase, _VariablesTest):
  dtype = tf.float32


class VariablesTestFloat64(tf.test.TestCase, _VariablesTest):
  dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
