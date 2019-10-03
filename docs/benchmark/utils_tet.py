import numpy as np
import tensorflow as tf

import utils
import t3f


class UtilsTest(tf.test.TestCase):

  def _TestCaseGrad(self, case):
    with self.session():
      tf.global_variables_initializer().run()
      tensors = []
      manual = case.naive_grad()
      try:
        manual_2 = case.smart_grad()
        self.assertAllClose(t3f.full(manual).eval(), t3f.full(manual_2).eval())
      except NotImplementedError:
        pass
      auto_g = t3f.gradients(case.loss, case.x, runtime_check=True)
      self.assertAllClose(t3f.full(manual).eval(), t3f.full(auto_g).eval(), rtol=1e-5)

  def _TestCaseHess(self, case):
    with self.session():
      tf.global_variables_initializer().run()
      manual = case.naive_hessian_by_vector()
      try:
        manual_2 = case.smart_hessian_by_vector()
        self.assertAllClose(t3f.full(manual).eval(), t3f.full(manual_2).eval())
      except NotImplementedError:
        pass
      auto_hv = t3f.hessian_vector_product(case.loss, case.x, case.vector, runtime_check=True)
      self.assertAllClose(t3f.full(manual).eval(), t3f.full(auto_hv).eval(), rtol=1e-5)

  def testCompletionGrad(self):
    test_case = utils.Completion(3, 3, 4)
    self._TestCaseGrad(test_case)

  def testCompletionHess(self):
    test_case = utils.Completion(3, 3, 4)
    self._TestCaseHess(test_case)

  def testXAXGrad(self):
    test_case = utils.BilinearXAX(3, 3, 3, 4, 5)
    self._TestCaseGrad(test_case)

  def testXAXHess(self):
    test_case = utils.BilinearXAX(3, 3, 3, 4, 5)
    self._TestCaseHess(test_case)

  def testXABXGrad(self):
    test_case = utils.BilinearXABX(3, 3, 3, 4, 5)
    self._TestCaseGrad(test_case)

  def testXABXHess(self):
    test_case = utils.BilinearXABX(3, 3, 3, 4, 5)
    self._TestCaseHess(test_case)

  def testExpMachinesGrad(self):
    test_case = utils.ExpMachines(3, 4, 5, batch_size=3)
    self._TestCaseGrad(test_case)

  def testExpMachinesHess(self):
    test_case = utils.ExpMachines(3, 3, 3, batch_size=2)
    self._TestCaseHess(test_case)

  def testRayleighQuotientGrad(self):
    test_case = utils.RayleighQuotient(3, 3, 3, 4, 5)
    self._TestCaseGrad(test_case)

  def testRayleighQuotientHess(self):
    test_case = utils.RayleighQuotient(3, 3, 3, 4, 5)
    self._TestCaseHess(test_case)


# class AutodiffTestFloat32(tf.test.TestCase, _AutodiffTest):
#   dtype = tf.float32


# class AutodiffTestFloat64(tf.test.TestCase, _AutodiffTest):
#   dtype = tf.float64


if __name__ == "__main__":
  tf.test.main()
