import numpy as np
import tensorflow as tf

import t3f
import utils

tf.reset_default_graph()
n = 400
d = 40
tt_rank_vec = 10
tt_rank_mat = 20
case = utils.BilinearXAX(n, n, d, tt_rank_mat, tt_rank_vec)
with tf.variable_scope("double"):
  double_case = utils.BilinearXAX(n, n, d, tt_rank_mat, tt_rank_vec*2)

naive_grad = case.naive_grad()
auto_grad = t3f.gradients(case.loss, case.x, runtime_check=False)
try:
  smart_grad = case.smart_grad()
except NotImplementedError:
  smart_grad = None
# Single configuration.
current_case_logs = {}
with tf.Session(config=tf.test.benchmark_config()) as sess:
  sess.run(tf.global_variables_initializer())
  benchmark = tf.test.Benchmark()
  current_case_logs['auto_grad'] = benchmark.run_op_benchmark(sess, auto_grad.op)
  current_case_logs['naive_grad'] = benchmark.run_op_benchmark(sess, naive_grad.op)
  current_case_logs['forward'] = benchmark.run_op_benchmark(sess, case.loss(case.x))
  current_case_logs['double_forward'] = benchmark.run_op_benchmark(sess, double_case.loss(double_case.x))  
  grad = tf.gradients(double_case.loss(double_case.x), double_case.x.tt_cores)
  current_case_logs['grad'] = benchmark.run_op_benchmark(sess, [g.op for g in grad])

print(current_case_logs)


from tensorflow.python.client import timeline
def profile(op, path):
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # add additional options to trace the session execution
      options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      sess.run(op)  # warmup
      sess.run(op, options=options, run_metadata=run_metadata)

      # Create the Timeline object, and write it to a json file
      fetched_timeline = timeline.Timeline(run_metadata.step_stats)
      chrome_trace = fetched_timeline.generate_chrome_trace_format()
      with open(path, 'w') as f:
          f.write(chrome_trace)
profile([c.op for c in auto_grad.tt_cores], 'auto.json')
profile([c.op for c in naive_grad.tt_cores], 'naive.json')
