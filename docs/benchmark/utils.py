import numpy as np
import tensorflow as tf
import numpy as np
import t3f
import json
import pickle


def robust_cumprod(arr):
  """Cumulative product with large values replaced by the MAX_DTYPE.
  
  robust_cumprod([10] * 100) = [10, 100, 1000, ..., MAX_INT, ..., MAX_INT] 
  """

  res = np.ones(arr.size, dtype=arr.dtype)
  change_large_to = np.iinfo(arr.dtype).max
  res[0] = arr[0]
  for i in range(1, arr.size):
    next_value = np.array(res[i - 1]) * np.array(arr[i])
    if next_value / np.array(arr[i]) != np.array(res[i - 1]):
      next_value = change_large_to
    res[i] = next_value
  return res


def max_tt_ranks(raw_shape):
  """Maximal TT-ranks for a TT-object of given shape.
  
  For example, a tensor of shape (2, 3, 5, 7) has maximal TT-ranks
    (1, 2, 6, 7, 1)
  making the TT-ranks larger will not increase flexibility.
  
  If maximum TT-ranks result in integer overflows, it substitutes
  the too-large-values with MAX_INT.
  
  Args:
    shape: an integer vector
  Returns:
    tt_ranks: an integer vector, maximal tt-rank for each dimension
  """
  raw_shape = np.array(raw_shape).astype(np.int64)
  d = raw_shape.size
  tt_ranks = np.zeros(d + 1, dtype='int64')
  tt_ranks[0] = 1
  tt_ranks[d] = 1
  left_to_right = robust_cumprod(raw_shape)
  right_to_left = robust_cumprod(raw_shape[::-1])[::-1]
  tt_ranks[1:-1] = np.minimum(left_to_right[:-1], right_to_left[1:])
  return tt_ranks

def sparse(idx, shape, dtype=None):
  cores = []
  for k in range(len(idx)):
    eye = tf.eye(shape[k], dtype=dtype)
    cores.append(tf.reshape(eye[idx[k]], (1, shape[k], 1)))
  return t3f.TensorTrain(cores)

def batch_sparse(idx_list, shape, weights=None, dtype=None):
  cores = []
  for k in range(len(idx_list[0])):
    curr_core = []
    eye = tf.eye(shape[k], dtype=dtype)
    cores.append(tf.reshape(tf.gather(eye, idx_list[:, k]), (-1, 1, shape[k], 1)))
  if weights is not None:
    cores[0] *= weights[:, None, None, None]
  return t3f.TensorTrainBatch(cores)


def reduce_sum_batch(x):
  tt_cores = list(x.tt_cores)
  for i, core in enumerate(tt_cores):
    bs, r1, n, r2 = core.shape.as_list()
    assert r1 == 1 and r2 == 1
    if i == 0:
      core = tf.reshape(core, (bs, 1, n))
      core = tf.transpose(core, (1, 2, 0))
    elif i == len(tt_cores) - 1:
      core = tf.reshape(core, (bs, n, 1))
    else:
      core = tf.tile(core[:, :, :, None, :], (1, 1, 1, bs, 1))
      core = tf.reshape(core, (bs, n, bs))
      core *= tf.tile(tf.eye(bs, dtype=x.dtype)[:, None, :], (1, n, 1))
    tt_cores[i] = core
  return t3f.TensorTrain(tt_cores)

def compare_tensors(tensors):
  for a in tensors:
    for b in tensors:
      a_np, b_np = sess.run([t3f.full(a), t3f.full(b)])
      diff = np.linalg.norm((a_np - b_np).flatten()) / np.linalg.norm(b_np)
      assert diff < 1e-8

def test(case, sess):
  tensors = []
  tensors.append(case.naive_grad())
  try:
    tensors.append(case.smart_grad())
  except NotImplementedError:
    pass
  auto_g = t3f.gradients(case.loss, case.x, runtime_check=True)
  tensors.append(auto_g)
  compare_tensors(tensors)
  
  tensors = []
  tensors.append(case.naive_hessian_by_vector())
  try:
    tensors.append(case.smart_hessian_by_vector())
  except NotImplementedError:
    pass
  auto_hv = t3f.hessian_vector_product(case.loss, case.x, case.vector, runtime_check=True)
  tensors.append(auto_hv)
  compare_tensors(tensors)
      


class Task(object):
  
  def smart_grad(self):
    return NotImplementedError()
  
  def naive_hessian_by_vector(self):
    return NotImplementedError()
  
  def smart_hessian_by_vector(self):
    return NotImplementedError()

    

class Completion(Task):
  
  def __init__(self, n, d, tt_rank):
    self.settings = {'n': n, 'd': d, 'tt_rank': tt_rank}
    shape = [n] * d
    self.num_observed = 10 * d * n * tt_rank**2  ###############################################################
    self.observation_idx = np.random.randint(0, n, size=(self.num_observed, len(shape)))
    self.observations_np = np.random.randn(self.num_observed)
    self.observations = tf.constant(self.observations_np)
    tt_rank_x = [1] + [tt_rank] * (d - 1) + [1]
    tt_rank_x = np.minimum(tt_rank_x, max_tt_ranks(shape))
    initialization = t3f.random_tensor(shape, tt_rank=tt_rank_x, dtype=tf.float64)
    self.x = t3f.get_variable('x', initializer=initialization)
    self.x *= 1.0 # Dtype bug
    tt_rank_v = [1] + [2 * tt_rank] * (d - 1) + [1]
    tt_rank_v = np.minimum(tt_rank_v, max_tt_ranks(shape))
    initialization = t3f.random_tensor(shape, tt_rank=tt_rank_v, dtype=tf.float64)
    self.vector = t3f.get_variable('vector', initializer=initialization)
    self.sparsity_mask_list_tt = batch_sparse(self.observation_idx, shape, dtype=tf.float64)
    self.sparsity_mask_tt = reduce_sum_batch(self.sparsity_mask_list_tt)
    self.sparse_observation_tt = reduce_sum_batch(batch_sparse(self.observation_idx, shape, self.observations_np, dtype=tf.float64))
    
  def loss(self, x):
    estimated_vals = t3f.gather_nd(x, self.observation_idx)
    return 0.5 * tf.reduce_sum((estimated_vals - self.observations_np) ** 2)
  
  def naive_grad(self):
    grad = self.sparsity_mask_tt * self.x - self.sparse_observation_tt
    return t3f.project(grad, self.x)
  
  def smart_grad(self):
    estimated_vals = t3f.gather_nd(self.x, self.observation_idx)
    diff = estimated_vals - self.observations
    return t3f.project_sum(self.sparsity_mask_list_tt, self.x, diff)
  
  def naive_hessian_by_vector(self):
    return t3f.project(self.sparsity_mask_tt * t3f.project(self.vector, self.x), self.x)
  
  def smart_hessian_by_vector(self):
    vector_nonzero = t3f.gather_nd(t3f.project(self.vector, self.x), self.observation_idx)
    return t3f.project_sum(self.sparsity_mask_list_tt, self.x, vector_nonzero)


def benchmark(case, prev_log=None):
  naive_grad = case.naive_grad()
  smart_grad = case.smart_grad()
  auto_grad = t3f.gradients(case.loss, case.x, runtime_check=False)

  naive_hv = case.naive_hessian_by_vector()
  smart_hv = case.smart_hessian_by_vector()
  auto_hv = t3f.hessian_vector_product(case.loss, case.x, case.vector, runtime_check=False)
  try:
    with open(r"logs.pickle", "rb") as output_file:
      all_logs_list = pickle.load(output_file)
  except:
    all_logs_list = []
  all_logs = {}
  with tf.Session(config=tf.test.benchmark_config()) as sess:
    sess.run(tf.global_variables_initializer())
    benchmark = tf.test.Benchmark()

    all_logs['settings'] = case.settings

    def benchmark_single(op, name, all_logs):
      try:
        if prev_log is not None and prev_log[name] is None:
          # No point in trying again, a smaller example failed already.
          raise ValueError()
        logs = benchmark.run_op_benchmark(sess, op)
        all_logs[name] = logs
      except:
        all_logs[name] = None

      with open(r"logs.pickle", "wb") as output_file:
        pickle.dump(all_logs_list + [all_logs], output_file)

    benchmark_single(auto_grad.op, 'auto_grad', all_logs)
    benchmark_single(auto_hv.op, 'auto_hv', all_logs)
    benchmark_single(smart_grad.op, 'smart_grad', all_logs)
    benchmark_single(smart_hv.op, 'smart_hv', all_logs)
    # benchmark_single(naive_grad.op, 'naive_grad', all_logs)
    # benchmark_single(naive_hv.op, 'naive_hv', all_logs)
    return all_logs
