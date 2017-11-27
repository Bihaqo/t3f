import timeit
import numpy as np

import tensorflow as tf
import t3f
import tt


def benchmark_tf(op, feed_dict, n):
  sess = tf.get_default_session()
  if sess is None:
    sess = tf.Session()

  best_time = np.inf
  for _ in range(n):
    start = timeit.default_timer()
    sess.run(op, feed_dict=feed_dict)
    end = timeit.default_timer()
    if end - start < best_time:
      best_time = end - start
  return best_time


def benchmark(t3f_build_op, tt_op, n=30):
  tt_time = np.inf
  for _ in range(n):
    start = timeit.default_timer()
    tt_op()
    end = timeit.default_timer()
    if end - start < tt_time:
      tt_time = end - start

  tf.reset_default_graph()
  add = tf.placeholder(tf.float32)
  with tf.device('cpu'):
    t3f_cpu_op = t3f_build_op(add)
  t3f_cpu_time = benchmark_tf(t3f_cpu_op, {add: 0.0}, n)

  tf.reset_default_graph()
  add = tf.placeholder(tf.float32)
  with tf.device('gpu'):
    t3f_gpu_op = t3f_build_op(add)
  try:
    t3f_gpu_time = benchmark_tf(t3f_gpu_op, {add: 0.0}, n)
  except:
    print('GPU device is not available.')
    t3f_gpu_time = None

  return tt_time, t3f_cpu_time, t3f_gpu_time

np.random.seed(0)

shape = ((8, 8, 8, 8), (8, 8, 8, 8))
tt_rank = 20
np_matrix = np.random.rand(2**12, 2**12).astype(np.float32)
add = tf.placeholder(tf.float32)


def t3f_build_to_matrix(add):
  return t3f.to_tt_matrix(np_matrix + add, shape, tt_rank).op


def tt_to_matrix():
  return tt.matrix(np_matrix + 0.0, n=shape[0], m=shape[1], rmax=tt_rank)

tt_time, t3f_cpu_time, t3f_gpu_time = benchmark(t3f_build_to_matrix,
                                                tt_to_matrix)
report = "Converting a matrix into a TT-matrix. TTPY takes %f s, " \
         "t3f on CPU %f s" % (tt_time, t3f_cpu_time)
if t3f_gpu_time is not None:
  report += ", t3f on GPU %f s." % t3f_gpu_time
print(report)
