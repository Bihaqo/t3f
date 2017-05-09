import timeit
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.client import device_lib
import t3f

print(device_lib.list_local_devices())


def my_timeit(f, number=100, repeat=3):
  # Warmup.
  for i in range(3):
    f()
  hist = []
  for i in range(number):
    current = np.inf
    for _ in range(repeat):
      s = timeit.default_timer()
      f()
      e = timeit.default_timer()
      current = min(current, e - s)
      hist.append(current)
  return np.mean(hist)

# Matvec.
shape = 10 * np.ones(10)
matrices = t3f.random_matrix_batch((shape, shape), 10, batch_size=100)
matrices = t3f.cast(matrices, tf.float64)
one_matrix = t3f.get_variable('one_matrix', initializer=matrices[0])
matrices = t3f.get_variable('matrices', initializer=matrices)
vecs = t3f.random_matrix_batch((shape, None), 10, batch_size=100)
vecs = t3f.cast(vecs, tf.float64)
one_vec = t3f.get_variable('one_vec', initializer=vecs[0])
vecs = t3f.get_variable('vecs', initializer=vecs)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
logs = {}

matvec_op = t3f.matmul(one_matrix, one_vec).op
matvec_time = my_timeit(lambda: sess.run(matvec_op))
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, one_vec,
                                                  matvec_time))
logs['matvec_time'] = matvec_time

batch_matvec_op = t3f.matmul(one_matrix, vecs).op
batch_matvec_time = my_timeit(lambda: sess.run(batch_matvec_op))
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, vecs,
                                                  batch_matvec_time))
logs['batch_matvec_time'] = batch_matvec_time

matmul_op = t3f.matmul(one_matrix, one_matrix).op
matmul_time = my_timeit(lambda: sess.run(matmul_op))
print('Multiplying %s by itself takes %f seconds.' % (one_matrix, matmul_time))
logs['matmul_time'] = matmul_time

batch_matmul_op = t3f.matmul(one_matrix, matrices).op
batch_matmul_time = my_timeit(lambda: sess.run(batch_matmul_op))
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, matrices,
                                                 batch_matmul_time))
logs['batch_matmul_time'] = batch_matmul_time

norm_op = t3f.frobenius_norm(one_matrix).op
norm_time = my_timeit(lambda: sess.run(norm_op))
print('Computing the norm of %s takes %f seconds.' % (one_matrix, norm_time))
logs['norm_time'] = norm_time

batch_norm_op = t3f.frobenius_norm(matrices).op
batch_norm_time = my_timeit(lambda: sess.run(batch_norm_op))
print('Computing the norm of %s takes %f seconds.' % (matrices, batch_norm_time))
logs['batch_norm_time'] = batch_norm_time

gram_op = t3f.gram_matrix(vecs).op
gram_time = my_timeit(lambda: sess.run(gram_op))
print('Computing the gram matrix of %s takes %f seconds.' % (vecs, gram_time))
logs['gram_time'] = gram_time

tens = tf.cast(tf.random_normal((10, 10, 10, 10)), tf.float64)
tt_svd_op = t3f.to_tt_tensor(tens, max_tt_rank=10).op
tt_svd_time = my_timeit(lambda: sess.run(tt_svd_op))
print('TT-SVD for tensor of shape %s takes %f seconds.' % (tens.get_shape(),
                                                           tt_svd_time))
logs['tt_svd_time'] = tt_svd_time

project_op = t3f.project(one_vec, one_vec).op
project_time = my_timeit(lambda: sess.run(project_op))
print('Projecting %s on %s takes %f seconds.' % (one_vec, one_vec, project_time))
logs['project_time'] = project_time

batch_project_op = t3f.project(vecs, one_vec).op
batch_project_time = my_timeit(lambda: sess.run(batch_project_op))
print('Projecting %s on %s takes %f seconds.' % (vecs, one_vec,
                                                 batch_project_time))
logs['batch_project_time'] = batch_project_time

pickle.save(open('profile_logs.pickle', 'wb'), logs)

