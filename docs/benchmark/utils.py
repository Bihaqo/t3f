import numpy as np
import tensorflow as tf
import numpy as np
import t3f
import json
import pickle
import copy


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
      

def prune_ranks(tt_rank, shape):
  tt_rank_arr = [1] + [tt_rank] * (len(shape) - 1) + [1]
  return np.minimum(tt_rank_arr, max_tt_ranks(shape))


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


class BilinearXAX(Task):
  
  def __init__(self, m, n, d, tt_rank_mat, tt_rank_vec):
    self.settings = {'n': n, 'm': m, 'd': d, 'tt_rank_mat': tt_rank_mat, 'tt_rank_vec': tt_rank_vec}
    shape = ([m] * d, [n] * d)
    ranks = prune_ranks(tt_rank_vec, shape[1])
    initialization = t3f.random_matrix((shape[1], None), tt_rank=ranks, dtype=tf.float64)
    self.x = t3f.get_variable('x', initializer=initialization)
    ranks = prune_ranks(2 * tt_rank_vec, shape[1])
    initialization = t3f.random_matrix((shape[1], None), tt_rank=ranks, dtype=tf.float64)
    self.vector = t3f.get_variable('vector', initializer=initialization)
    ranks = prune_ranks(tt_rank_mat, np.prod(shape, axis=0))
    mat = t3f.random_matrix(shape, tt_rank=ranks, dtype=tf.float64)
    mat = t3f.transpose(mat) + mat
    self.mat = t3f.get_variable('mat', initializer=mat)
    
  def loss(self, x):
    return 0.5 * t3f.quadratic_form(self.mat, x, x)  # DO NOT SUBMIT
  
  def naive_grad(self):
    grad = t3f.matmul(self.mat, self.x)  # DO NOT SUBMIT
    return t3f.project(grad, self.x)
  
  def smart_grad(self):
    return t3f.project_matmul(t3f.expand_batch_dim(self.x), self.x, self.mat)[0]  # DO NOT SUBMIT
  
  def naive_hessian_by_vector(self):
    grad = t3f.matmul(self.mat, self.vector)
    return t3f.project(grad, self.x)
  
  def smart_hessian_by_vector(self):
    return t3f.project_matmul(t3f.expand_batch_dim(self.vector), self.x, self.mat)[0]


class ExpMachines(Task):
  
  def __init__(self, n, d, tt_rank_vec, batch_size=32):
    self.settings = {'n': n, 'd': d, 'tt_rank_vec': tt_rank_vec}
    shape = [n] * d
    ranks = prune_ranks(tt_rank_vec, shape)
    initialization = t3f.random_tensor(shape, tt_rank=ranks, dtype=tf.float64)
    self.x = t3f.get_variable('x', initializer=initialization)
    initialization = t3f.random_tensor_batch(shape, tt_rank=1, dtype=tf.float64, batch_size=batch_size)
    self.w = t3f.get_variable('w', initializer=initialization)
    ranks = prune_ranks(2 * tt_rank_vec, shape)
    initialization = t3f.random_tensor(shape, tt_rank=ranks, dtype=tf.float64)
    self.vector = t3f.get_variable('vector', initializer=initialization)
    
  def loss(self, x):
    l = t3f.flat_inner(x, self.w)
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=l, labels=tf.ones(self.w.batch_size, dtype=tf.float64)))
  
  def naive_grad(self):
    e = tf.exp(-1. * t3f.flat_inner(self.x, self.w))
    c = -e / (1 + e)
    grad = c[0] * self.w[0]
    for i in range(1, self.w.batch_size):
      grad += c[i] * self.w[i]
    return t3f.project(grad, self.x)
  
  def smart_grad(self):
    e = tf.exp(-1. * t3f.flat_inner(self.x, self.w))
    c = -e / (1 + e)
    return t3f.project_sum(self.w, 1. * self.x, c)
  
  def naive_hessian_by_vector(self):
    e = tf.exp(-1. * t3f.flat_inner(self.x, self.w))
    s = 1. / (1 + e)
    c = s * (1 - s)
    c *= t3f.flat_inner(self.vector, self.w)
    res = c[0] * self.w[0]
    for i in range(1, self.w.batch_size):
      res += c[i] * self.w[i]
    return t3f.project(res, self.x)
  
  def smart_hessian_by_vector(self):
    e = tf.exp(-1. * t3f.flat_inner(self.x, self.w))
    s = 1. / (1 + e)
    c = s * (1 - s)
    c *= t3f.flat_inner(self.vector, self.w)
    return t3f.project_sum(self.w, 1. * self.x, c)


class BilinearXABX(Task):
  
  def __init__(self, m, n, d, tt_rank_mat, tt_rank_vec):
    self.settings = {'n': n, 'm': m, 'd': d, 'tt_rank_mat': tt_rank_mat, 'tt_rank_vec': tt_rank_vec}
    shape = ([m] * d, [n] * d)
    ranks = prune_ranks(tt_rank_vec, shape[1])
    initialization = t3f.random_matrix((shape[1], None), tt_rank=ranks, dtype=tf.float64)
    self.x = t3f.get_variable('x', initializer=initialization)
    ranks = prune_ranks(2 * tt_rank_vec, shape[1])
    initialization = t3f.random_matrix((shape[1], None), tt_rank=ranks, dtype=tf.float64)
    self.vector = t3f.get_variable('vector', initializer=initialization)
    ranks = prune_ranks(tt_rank_mat, np.prod(shape, axis=0))
    initialization = t3f.random_matrix(shape, tt_rank=ranks, dtype=tf.float64)
    self.mat = t3f.get_variable('mat', initializer=initialization)
    
  def loss(self, x):
    return 0.5 * t3f.bilinear_xaby(x, t3f.transpose(self.mat), self.mat, x)
  
  def naive_grad(self):
    grad = t3f.matmul(t3f.transpose(self.mat), t3f.matmul(self.mat, self.x))
    return t3f.project(grad, self.x)
  
  def smart_grad(self):
    raise NotImplementedError()
  
  def naive_hessian_by_vector(self):
    projected_vec = t3f.project(self.vector, self.x)
    return t3f.project(t3f.matmul(t3f.transpose(self.mat), t3f.matmul(self.mat, projected_vec)), self.x)
  
  def smart_hessian_by_vector(self):
    raise NotImplementedError()


class RayleighQuotient(Task):
  
  def __init__(self, m, n, d, tt_rank_mat, tt_rank_vec):
    self.settings = {'n': n, 'm': m, 'd': d, 'tt_rank_mat': tt_rank_mat, 'tt_rank_vec': tt_rank_vec}
    shape = ([m] * d, [n] * d)
    ranks = prune_ranks(tt_rank_vec, shape[1])
    initialization = t3f.random_matrix((shape[1], None), tt_rank=ranks, dtype=tf.float64)
    self.x = t3f.get_variable('x', initializer=initialization)
    ranks = prune_ranks(2 * tt_rank_vec, shape[1])
    initialization = t3f.random_matrix((shape[1], None), tt_rank=ranks, dtype=tf.float64)
    self.vector = t3f.get_variable('vector', initializer=initialization)
    ranks = prune_ranks(tt_rank_mat, np.prod(shape, axis=0))
    mat = t3f.random_matrix(shape, tt_rank=ranks, dtype=tf.float64)
    mat = t3f.transpose(mat) + mat
    self.mat = t3f.get_variable('mat', initializer=mat)
    
  def loss(self, x):
    xAx = t3f.quadratic_form(self.mat, x, x)  # bilinear_form
    xx = t3f.flat_inner(x, x)
    return xAx / xx
  
  def naive_grad(self):
    xAx = t3f.quadratic_form(self.mat, self.x, self.x)  # bilinear_form
    xx = t3f.flat_inner(self.x, self.x)
    grad = (1. / xx) * t3f.matmul(self.mat, self.x)
    grad -= (xAx / (xx**2)) * self.x
    return t3f.project(2 * grad, self.x)
  
  def smart_grad(self):
    xAx = t3f.quadratic_form(self.mat, self.x, self.x)  # bilinear_form
    xx = t3f.frobenius_norm_squared(self.x, differentiable=True)
    grad = (1. / xx) * t3f.project_matmul(t3f.expand_batch_dim(self.x), self.x, self.mat)[0]
    grad -= (xAx / xx**2) * self.x
    return 2 * grad
  
  def naive_hessian_by_vector(self):
    xAx = t3f.quadratic_form(self.mat, self.x, self.x)  # bilinear_form
    xx = t3f.frobenius_norm_squared(self.x, differentiable=True)
    res = (2 / xx) * t3f.matmul(self.mat, self.vector)
    res -= (2 * xAx / xx**2) * self.vector
    xv = t3f.flat_inner(self.x, self.vector)
    res -= (4 * t3f.quadratic_form(self.mat, self.vector, self.x) / xx**2) * self.x
    res -= (4 * xv / xx**2) * t3f.matmul(self.mat, self.x)
    res += (8 * xAx * xv / xx**3) * self.x
    return t3f.project(res, self.x)
  
  def smart_hessian_by_vector(self):
    xAx = t3f.quadratic_form(self.mat, self.x, self.x)  # bilinear_form
    xx = t3f.frobenius_norm_squared(self.x, differentiable=True)
    projected_vec = t3f.project(self.vector, self.x)
    res = (2 / xx) * t3f.project_matmul(t3f.expand_batch_dim(self.vector), self.x, self.mat)[0]
    res -= (2 * xAx / xx**2) * projected_vec
    xv = t3f.flat_inner(self.x, projected_vec)
    res -= (4 * t3f.quadratic_form(self.mat, self.vector, self.x) / xx**2) * self.x
    res -= (4 * xv / xx**2) * t3f.project_matmul(t3f.expand_batch_dim(self.x), self.x, self.mat)[0]
    res += (8 * xAx * xv / xx**3) * self.x
    return res


def exist(all_logs, case_name, case):
  for l in all_logs[case_name]:
    s = l['settings']
    coincide = True
    for k in case.settings:
      if s[k] != case.settings[k]:
        coincide = False
    if coincide:
      return True
  return False


def did_smaller_fail(all_logs, name, case_name, case):
  for l in all_logs[case_name]:
    s = l['settings']
    if name in l and l[name] is None:
      # If this attempt failed.
      smaller = True
      for k in case.settings:
        if s[k] > case.settings[k]:
          smaller = False
      if smaller:
        return True
  return False


def benchmark(case_name, case, logs_path):
  naive_grad = case.naive_grad()
  auto_grad = t3f.gradients(case.loss, case.x, runtime_check=False)
  try:
    smart_grad = case.smart_grad()
  except NotImplementedError:
    smart_grad = None

  naive_hv = case.naive_hessian_by_vector()
  auto_hv = t3f.hessian_vector_product(case.loss, case.x, case.vector, runtime_check=False)
  try:
    smart_hv = case.smart_hessian_by_vector()
  except NotImplementedError:
    smart_hv = None
  try:
    with open(logs_path, "rb") as output_file:
      # Dict with case_name -> list of configurations.
      all_logs = pickle.load(output_file)
  except:
    all_logs = {}
  if case_name not in all_logs:
    all_logs[case_name] = []
  # Single configuration.
  current_case_logs = {}
  with tf.Session(config=tf.test.benchmark_config()) as sess:
    sess.run(tf.global_variables_initializer())
    benchmark = tf.test.Benchmark()

    current_case_logs['settings'] = case.settings
    if exist(all_logs, case_name, case):
      print('skipping')
      return None


    def benchmark_single(op, name, current_case_logs):
      # First write None to indicate the attempt.
      with open(logs_path, "wb") as output_file:
        all_logs_curr = copy.deepcopy(all_logs)
        current_case_logs[name] = None
        all_logs_curr[case_name].append(current_case_logs)
        pickle.dump(all_logs_curr, output_file)

      try:
        if did_smaller_fail(all_logs, name, case_name, case):
          # No point in trying again, a smaller example failed already.
          raise ValueError()
        logs = benchmark.run_op_benchmark(sess, op)
        current_case_logs[name] = logs
      except:
        current_case_logs[name] = None

      with open(logs_path, "wb") as output_file:
        all_logs_curr = copy.deepcopy(all_logs)
        all_logs_curr[case_name].append(current_case_logs)
        pickle.dump(all_logs_curr, output_file)

    benchmark_single(auto_grad.op, 'auto_grad', current_case_logs)
    benchmark_single(auto_hv.op, 'auto_hv', current_case_logs)
    if smart_grad is not None:
      benchmark_single(smart_grad.op, 'smart_grad', current_case_logs)
    if smart_hv is not None:
      benchmark_single(smart_hv.op, 'smart_hv', current_case_logs)
    benchmark_single(naive_grad.op, 'naive_grad', current_case_logs)
    benchmark_single(naive_hv.op, 'naive_hv', current_case_logs)
    return current_case_logs
