import timeit
import numpy as np
import pickle
import argparse
import tensorflow as tf


# TODO: remove this after the next release of TF (which should include
# tf.test.benchmark_config())
try:
  tf.test.benchmark_config()
except AttributeError:
  from tensorflow import core
  def benchmark_config():
    """Returns a tf.ConfigProto for disabling the dependency optimizer.
      Returns:
        A TensorFlow ConfigProto object.
    """
    config = core.protobuf.config_pb2.ConfigProto()
    config.graph_options.rewrite_options.dependency_optimization = (
      core.protobuf.rewriter_config_pb2.RewriterConfig.OFF)
    return config
  tf.test.benchmark_config = benchmark_config


from tensorflow.python.client import device_lib
import t3f

parser = argparse.ArgumentParser(description='Measure execution time of various t3f operations.')
parser.add_argument('--file_path', help='Path to the file to save logs.')
args = parser.parse_args()


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
sess = tf.Session(config=tf.test.benchmark_config())
sess.run(tf.global_variables_initializer())
print(device_lib.list_local_devices())
logs = {}

matvec_op = t3f.matmul(one_matrix, one_vec).op
benchmark = tf.test.Benchmark()
logs['matvec'] = benchmark.run_op_benchmark(sess, matvec_op)
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, one_vec,
                                                  logs['matvec']['wall_time']))

batch_matvec_op = t3f.matmul(one_matrix, vecs).op
logs['batch_matvec'] = benchmark.run_op_benchmark(sess, batch_matvec_op)
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, vecs,
                                                  logs['batch_matvec']['wall_time']))

matmul_op = t3f.matmul(one_matrix, one_matrix).op
logs['matmul'] = benchmark.run_op_benchmark(sess, matmul_op)
print('Multiplying %s by itself takes %f seconds.' % (one_matrix, logs['matmul']['wall_time']))

batch_matmul_op = t3f.matmul(one_matrix, matrices).op
logs['batch_matmul'] = benchmark.run_op_benchmark(sess, batch_matmul_op)
print('Multiplying %s by %s takes %f seconds.' % (one_matrix, matrices,
                                                 logs['batch_matmul']['wall_time']))

norm_op = t3f.frobenius_norm(one_matrix, differentiable=True).op
logs['norm'] = benchmark.run_op_benchmark(sess, norm_op)
print('Computing the norm of %s takes %f seconds.' % (one_matrix, logs['norm']['wall_time']))

batch_norm_op = t3f.frobenius_norm(matrices, differentiable=True).op
logs['batch_norm'] = benchmark.run_op_benchmark(sess, batch_norm_op)
print('Computing the norm of %s takes %f seconds.' % (matrices, logs['batch_norm']['wall_time']))

flatinner_op = t3f.flat_inner(one_vec, one_vec).op
logs['flatinner'] = benchmark.run_op_benchmark(sess, flatinner_op)
print('Computing the dot product between %s and itself takes %f seconds.' %
      (one_vec, logs['flatinner']['wall_time']))

gram_op = t3f.gram_matrix(vecs).op
logs['batch_gram'] = benchmark.run_op_benchmark(sess, gram_op)
print('Computing the gram matrix of %s takes %f seconds.' % (vecs, logs['batch_gram']['wall_time']))

tens = tf.cast(tf.random_normal((10, 10, 10, 10)), tf.float64)
tt_svd_op = t3f.to_tt_tensor(tens, max_tt_rank=10).op
logs['tt_svd'] = benchmark.run_op_benchmark(sess, tt_svd_op)
print('TT-SVD for tensor of shape %s takes %f seconds.' % (tens.get_shape(),
                                                           logs['tt_svd']['wall_time']))

round_op = t3f.round(one_vec100, max_tt_rank=10).op
logs['round'] = benchmark.run_op_benchmark(sess, round_op)
print('Rounding %s takes %f seconds.' % (one_vec100, logs['round']['wall_time']))

batch_round_op = t3f.round(vecs100, max_tt_rank=10).op
logs['batch_round'] = benchmark.run_op_benchmark(sess, batch_round_op)
print('Rounding %s takes %f seconds.' % (vecs100, logs['batch_round']['wall_time']))

project_op = t3f.project(one_vec, one_vec).op
logs['project'] = benchmark.run_op_benchmark(sess, project_op)
print('Projecting %s on %s takes %f seconds.' % (one_vec, one_vec, logs['project']['wall_time']))

batch_project_op = t3f.project(vecs, one_vec).op
logs['batch_project'] = benchmark.run_op_benchmark(sess, batch_project_op)
print('Projecting %s on %s takes %f seconds.' % (vecs, one_vec,
                                                 logs['batch_project']['wall_time']))

project100_op = t3f.project(one_vec100, one_vec).op
logs['project_rank100'] = benchmark.run_op_benchmark(sess, project100_op)
print('Projecting %s on %s takes %f seconds.' % (one_vec100, one_vec, logs['project_rank100']['wall_time']))

batch_project100_op = t3f.project(vecs100, one_vec).op
logs['batch_project_rank100'] = benchmark.run_op_benchmark(sess, batch_project100_op)
print('Projecting %s on %s takes %f seconds.' % (vecs100, one_vec,
  logs['batch_project_rank100']['wall_time']))

if args.file_path is not None:
  pickle.dump(logs, open(args.file_path, 'wb'))

