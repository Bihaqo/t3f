import timeit
import numpy as np
import pickle
import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
import t3f

parser = argparse.ArgumentParser(description='Measure execution time of various t3f operations.')
parser.add_argument('--file_path', help='Path to the file to save logs.')
args = parser.parse_args()

print(device_lib.list_local_devices())

# Matvec.
shape = 10 * np.ones(10, dtype=int)
matrices = t3f.random_matrix_batch((shape, shape), 10, batch_size=100)
matrices = t3f.cast(matrices, tf.float64)
one_matrix = t3f.get_variable('one_matrix', initializer=matrices[0])
matrices = t3f.get_variable('matrices', initializer=matrices)
vecs = t3f.random_matrix_batch((shape, None), 10, batch_size=100)
vecs = t3f.cast(vecs, tf.float64)
one_vec = t3f.get_variable('one_vec', initializer=vecs[0])
vecs = t3f.get_variable('vecs', initializer=vecs)
vecs100 = t3f.random_matrix_batch((shape, None), 100, batch_size=100)
vecs100 = t3f.cast(vecs100, tf.float64)
one_vec100 = t3f.get_variable('one_vec100', initializer=vecs100[0])
vecs100 = t3f.get_variable('vecs100', initializer=vecs100)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
logs = {}

matvec_op = t3f.matmul(one_matrix, one_vec).op
# Warmup.
timeit.timeit("sess.run(matvec_op)",
              globals={'sess': sess, 'matvec_op': matvec_op},
              number=10)
matvec_time = timeit.timeit("sess.run(matvec_op)",
                            globals={'sess': sess, 'matvec_op': matvec_op},
                            number=1000) / 1000
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, one_vec,
                                                  matvec_time))
logs['matvec_time'] = matvec_time

batch_matvec_op = t3f.matmul(one_matrix, vecs).op
batch_matvec_time = timeit.timeit("sess.run(batch_matvec_op)",
                            globals={'sess': sess, 'batch_matvec_op': batch_matvec_op},
                            number=100) / 100
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, vecs,
                                                  batch_matvec_time))
logs['batch_matvec_time'] = batch_matvec_time

matmul_op = t3f.matmul(one_matrix, one_matrix).op
matmul_time = timeit.timeit("sess.run(matmul_op)",
                            globals={'sess': sess, 'matmul_op': matmul_op},
                            number=1000) / 1000
print('Multiplying %s by itself takes %f seconds.' % (one_matrix, matmul_time))
logs['matmul_time'] = matmul_time

batch_matmul_op = t3f.matmul(one_matrix, matrices).op
batch_matmul_time = timeit.timeit("sess.run(batch_matmul_op)",
                            globals={'sess': sess, 'batch_matmul_op': batch_matmul_op},
                            number=100) / 100
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, matrices,
                                                 batch_matmul_time))
logs['batch_matmul_time'] = batch_matmul_time

norm_op = t3f.frobenius_norm(one_matrix, differentiable=True).op
norm_time = timeit.timeit("sess.run(norm_op)",
                            globals={'sess': sess, 'norm_op': norm_op},
                            number=1000) / 1000
print('Computing the norm of %s takes %f seconds.' % (one_matrix, norm_time))
logs['norm_time'] = norm_time

batch_norm_op = t3f.frobenius_norm(matrices, differentiable=True).op
batch_norm_time = timeit.timeit("sess.run(batch_norm_op)",
                            globals={'sess': sess, 'batch_norm_op': batch_norm_op},
                            number=1000) / 1000
print('Computing the norm of %s takes %f seconds.' % (matrices, batch_norm_time))
logs['batch_norm_time'] = batch_norm_time

flatinner_op = t3f.flat_inner(one_vec, one_vec).op
flatinner_time = timeit.timeit("sess.run(flatinner_op)",
                            globals={'sess': sess, 'flatinner_op': flatinner_op},
                            number=1000) / 1000
print('Computing the dot product between %s and itself takes %f seconds.' %
      (one_vec, flatinner_time))
logs['flatinner_time'] = flatinner_time

gram_op = t3f.gram_matrix(vecs).op
gram_time = timeit.timeit("sess.run(gram_op)",
                            globals={'sess': sess, 'gram_op': gram_op},
                            number=100) / 100
print('Computing the gram matrix of %s takes %f seconds.' % (vecs, gram_time))
logs['batch_gram_time'] = gram_time

tens = tf.cast(tf.random_normal((10, 10, 10, 10)), tf.float64)
tt_svd_op = t3f.to_tt_tensor(tens, max_tt_rank=10).op
tt_svd_time = timeit.timeit("sess.run(tt_svd_op)",
                            globals={'sess': sess, 'tt_svd_op': tt_svd_op},
                            number=1000) / 1000
print('TT-SVD for tensor of shape %s takes %f seconds.' % (tens.get_shape(),
                                                           tt_svd_time))
logs['tt_svd_time'] = tt_svd_time

round_op = t3f.round(one_vec100, max_tt_rank=10).op
round_time = timeit.timeit("sess.run(round_op)",
                            globals={'sess': sess, 'round_op': round_op},
                            number=1000) / 1000
print('Rounding %s takes %f seconds.' % (one_vec100, round_time))
logs['round_time'] = round_time

batch_round_op = t3f.round(vecs100, max_tt_rank=10).op
batch_round_time = timeit.timeit("sess.run(batch_round_op)",
                            globals={'sess': sess, 'batch_round_op': batch_round_op},
                            number=100) / 100
print('Rounding %s takes %f seconds.' % (vecs100, batch_round_time))
logs['batch_round_time'] = batch_round_time

project_op = t3f.project(one_vec, one_vec).op
project_time = timeit.timeit("sess.run(project_op)",
                            globals={'sess': sess, 'project_op': project_op},
                            number=1000) / 1000
print('Projecting %s on %s takes %f seconds.' % (one_vec, one_vec, project_time))
logs['project_time'] = project_time

batch_project_op = t3f.project(vecs, one_vec).op
batch_project_time = timeit.timeit("sess.run(batch_project_op)",
                            globals={'sess': sess, 'batch_project_op': batch_project_op},
                            number=100) / 100
print('Projecting %s on %s takes %f seconds.' % (vecs, one_vec,
                                                 batch_project_time))
logs['batch_project_time'] = batch_project_time

project100_op = t3f.project(one_vec100, one_vec).op
project100_time = timeit.timeit("sess.run(project100_op)",
                            globals={'sess': sess, 'project100_op': project100_op},
                            number=1000) / 1000
print('Projecting %s on %s takes %f seconds.' % (one_vec100, one_vec, project100_time))
logs['project_rank100_time'] = project100_time

batch_project100_op = t3f.project(vecs100, one_vec).op
batch_project100_time = timeit.timeit("sess.run(batch_project100_op)",
                            globals={'sess': sess, 'batch_project100_op': batch_project100_op},
                            number=1000) / 1000
print('Projecting %s on %s takes %f seconds.' % (vecs100, one_vec, batch_project100_time))
logs['batch_project_rank100_time'] = batch_project100_time

if args.file_path is not None:
  pickle.dump(logs, open(args.file_path, 'wb'))

