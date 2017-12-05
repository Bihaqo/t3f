import tensorflow as tf

from t3f import ops
from t3f import approximate
from t3f import initializers


class ApproximateTest(tf.test.TestCase):

  def testAddN(self):
    # Sum a bunch of TT-matrices.
    tt_a = initializers.random_matrix(((2, 1, 4), (2, 2, 2)), tt_rank=2)
    tt_b = initializers.random_matrix(((2, 1, 4), (2, 2, 2)),
                                            tt_rank=[1, 2, 4, 1])

    def desired(tt_objects):
      res = tt_objects[0]
      for tt in tt_objects[1:]:
        res += tt
      return res

    with self.test_session() as sess:
      res_actual = ops.full(approximate.add_n([tt_a, tt_b], 6))
      res_desired = ops.full(desired([tt_a, tt_b]))
      res_desired_val, res_actual_val = sess.run([res_desired, res_actual])
      self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

      res_actual = ops.full(approximate.add_n([tt_a, tt_b, tt_a], 8))
      res_desired = ops.full(desired([tt_a, tt_b, tt_a]))
      res_desired_val, res_actual_val = sess.run([res_desired, res_actual])
      self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)

      res_actual = ops.full(approximate.add_n([tt_a, tt_b, tt_a, tt_a, tt_a], 12))
      res_desired = ops.full(desired([tt_a, tt_b, tt_a, tt_a, tt_a]))
      res_desired_val, res_actual_val = sess.run([res_desired, res_actual])
      self.assertAllClose(res_desired_val, res_actual_val, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
