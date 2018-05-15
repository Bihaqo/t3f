import tensorflow as tf

from t3f.tensor_train import TensorTrain
from t3f import shapes
from t3f import batch_ops
from t3f import decompositions
from t3f import riemannian


# TODO: move to riemannian.
def tangent_space_to_deltas(tt, name='t3f_tangent_space_to_deltas'):
  """Convert an element of the tangent space to deltas representation.

  Tangent space elements (outputs of t3f.project) look like:
    dP1 V2 ... Vd + U1 dP2 V3 ... Vd + ... + U1 ... Ud-1 dPd.

  This function takes as input an element of the tangent space and converts
  it to the list of deltas [dP1, ..., dPd].

  Args:
      tt: `TensorTrain` or `TensorTrainBatch` that is a result of t3f.project,
        t3f.project_matmul, or other similar functions.
      name: string, name of the Op.

  Returns:
      A list of delta-cores.
  """
  if not hasattr(tt, 'projection_on') or tt.projection_on is None:
    raise ValueError('tt argument is supposed to be a projection, but it '
                     'lacks projection_on field')
  num_dims = tt.ndims()
  left_tt_rank_dim = tt.left_tt_rank_dim
  right_tt_rank_dim = tt.right_tt_rank_dim
  deltas = [None] * num_dims
  tt_ranks = shapes.lazy_tt_ranks(tt)
  for i in range(1, num_dims - 1):
      if int(tt_ranks[i] / 2) != tt_ranks[i] / 2:
        raise ValueError('tt argument is supposed to be a projection, but its '
                         'ranks are not even.')
  with tf.name_scope(name, values=tt.tt_cores):
    for i in range(1, num_dims - 1):
      r1, r2 = tt_ranks[i], tt_ranks[i + 1]
      curr_core = tt.tt_cores[i]
      slc = [slice(None)] * len(curr_core.shape)
      slc[left_tt_rank_dim] = slice(int(r1 / 2), None)
      slc[right_tt_rank_dim] = slice(0, int(r2 / 2))
      deltas[i] = curr_core[slice]
    slc = [slice(None)] * len(tt.tt_cores[0].shape)
    slc[right_tt_rank_dim] = slice(0, int(tt_ranks[1] / 2))
    deltas[0] = tt.tt_cores[0][slc]
    slc = [slice(None)] * len(tt.tt_cores[0].shape)
    slc[left_tt_rank_dim] = slice(int(tt_ranks[-1] / 2), None)
    deltas[num_dims - 1] = tt.tt_cores[num_dims - 1][slc]
  return deltas


# TODO: move to riemannian.
def deltas_to_tangent_space(deltas, tt, left=None, right=None,
                            name='t3f_deltas_to_tangent_space'):
  """Converts deltas representation of tangent space vector to TensorTrain object.

  Takes as input a list of [dP1, ..., dPd] and returns
    dP1 V2 ... Vd + U1 dP2 V3 ... Vd + ... + U1 ... Ud-1 dPd.

  This function is hard to use correctly because deltas should abey the
  so called gauge conditions. If the don't, the function will silently return
  incorrect result. This is why this function is not imported in the __init__.

  Args:
      deltas: a list of deltas (essentially TT-cores).
      tt: `TensorTrain` object on which the tangent space tensor represented by
        delta is projected.
      left: t3f.orthogonilize_tt_cores(tt). If you have it already compute, you
        may pass it as argument to avoid recomputing.
      right: t3f.orthogonilize_tt_cores(left, left_to_right=False). If you have
        it already compute, you may pass it as argument to avoid recomputing.
      name: string, name of the Op.

  Returns:
      `TensorTrain` object constructed from deltas, that is from the tangent
        space at point `tt`.
  """
  cores = []
  dtype = deltas[0].dtype
  num_dims = left.ndims()
  # TODO: add cache instead of mannually pasisng precomputed stuff?
  if left is None:
    left = decompositions.orthogonalize_tt_cores(tt)
  if right is None:
    right = decompositions.orthogonalize_tt_cores(left, left_to_right=False)
  left_tangent_tt_ranks = shapes.lazy_tt_ranks(left)
  right_tangent_tt_ranks = shapes.lazy_tt_ranks(left)
  raw_shape = shapes.lazy_raw_shape(left)
  right_rank_dim = left.right_tt_rank_dim
  left_rank_dim = left.left_tt_rank_dim
  is_batch_case = hasattr(tt, 'batch_size')
  for i in range(num_dims):
    left_tt_core = left.tt_cores[i]
    right_tt_core = right.tt_cores[i]

    if i == 0:
      tangent_core = tf.concat((deltas[i], left_tt_core),
                               axis=right_rank_dim)
    elif i == num_dims - 1:
      tangent_core = tf.concat((right_tt_core, deltas[i]),
                               axis=left_rank_dim)
    else:
      rank_1 = right_tangent_tt_ranks[i]
      rank_2 = left_tangent_tt_ranks[i + 1]
      if tt.is_tt_matrix():
        mode_size_n = raw_shape[0][i]
        mode_size_m = raw_shape[1][i]
        shape = [rank_1, mode_size_n, mode_size_m, rank_2]
      else:
        mode_size_n = raw_shape[0][i]
        shape = [rank_1, mode_size_n, rank_2]
      if is_batch_case:
        shape = [tt.batch_size] + shape
      zeros = tf.zeros(shape, dtype=dtype)
      upper = tf.concat((right_tt_core, zeros), axis=right_rank_dim)
      lower = tf.concat((deltas[i], left_tt_core), axis=right_rank_dim)
      tangent_core = tf.concat((upper, lower), axis=left_rank_dim)
    cores.append(tangent_core)
  # Create instance of the same class as tt.
  tangent = tt.__class__(cores)
  tangent.projection_on = tt
  return tangent


def _gradients(func, x, x_projection, left, right):
  """Internal version of t3f.gradients that assumes some precomputed inputs."""
  h = func(x_projection)
  cores_grad = tf.gradients(h, x_projection.tt_cores)
  deltas = []
  for i in range(x.ndims()):
    if x.is_tt_matrix():
      r1, n, m, r2 = left.tt_cores[i].shape.as_list()
    else:
      r1, n, r2 = left.tt_cores[i].shape.as_list()
    q = tf.reshape(left.tt_cores[i], (-1, r2))
    if x.is_tt_matrix():
      if i == 0:
        curr_grad = cores_grad[i][:, :, :, :r2]
      elif i == x.ndims() - 1:
        curr_grad = cores_grad[i][r1:, :, :, :]
      else:
        curr_grad = cores_grad[i][r1:, :, :, :r2]
    else:
      if i == 0:
        curr_grad = cores_grad[i][:, :, :r2]
      elif i == w.ndims() - 1:
        curr_grad = cores_grad[i][r1:, :, :]
      else:
        curr_grad = cores_grad[i][r1:, :, :r2]
    if i < x.ndims() - 1:
      delta = curr_grad
      delta = tf.reshape(delta, (-1, r2))
      delta -= tf.matmul(q, tf.matmul(tf.transpose(q), delta))
      delta = tf.reshape(delta, left.tt_cores[i].shape)
    else:
      delta = curr_grad
    deltas.append(delta)
  return deltas_to_tangent_space(deltas, x, left, right)


def gradients(func, x, name='t3f_gradients'):
  """Riemannian autodiff: returns gradient projected on tangent space of TT.

  Automatically computes projection of the gradient df/dx onto the
  tangent space of TT tensor at point x.

  Warning: this is experimental feature and it may not work for some function,
  e.g. ones that include QR or SVD decomposition (t3f.project, t3f.round) or
  for functions that work with TT-cores directly (in contrast to working with
  TT-object only via t3f functions). In this cases this function can silently
  return wrong results!

  Example:
      # Scalar product with some predefined tensor squared 0.5 * <x, t>**2.
      # It's gradient is <x, t> t and it's Riemannian gradient is
      #     t3f.project(<x, t> * t, x)
      f = lambda x: 0.5 * t3f.flat_inner(x, t)**2
      projected_grad = t3f.gradients(f, x) # t3f.project(t3f.flat_inner(x, t) * t, x)

  Args:
      func: function that takes TensorTrain object as input and outputs a number.
      x: point at which to compute the gradient and on which tangent space to
        project the gradient.
      name: string, name of the Op.

  Returns:
      `TensorTrain`, projection of the gradient df/dx onto the tangent space at
      point x.

  See also:
      t3f.hessian_vector_product
  """
  with tf.name_scope(name, values=x.tt_cores):
    left = decompositions.orthogonalize_tt_cores(x)
    right = decompositions.orthogonalize_tt_cores(left, left_to_right=False)
    deltas = [right.tt_cores[0]] + [tf.zeros_like(cc) for cc in right.tt_cores[1:]]
    x_projection = deltas_to_tangent_space(deltas, x, left, right)
    return _gradients(func, x, x_projection, left, right)


def hessian_vector_product(func, x, vector, name='t3f_hessian_vector_product'):
  """P_x d^2f/dx^2 P_x vector, i.e. Riemannian hessian by vector product.

    Automatically computes
      P_x d^2f/dx^2 P_x vector
    where P_x is projection onto the tangent space of TT at point x and
    d^2f/dx^2 is the Hessian of the function.

    Note that the true hessian also includes the manifold curvature term
    which is ignored here.

    Warning: this is experimental feature and it may not work for some function,
    e.g. ones that include QR or SVD decomposition (t3f.project, t3f.round) or
    for functions that work with TT-cores directly (in contrast to working with
    TT-object only via t3f functions). In this cases this function can silently
    return wrong results!

    Example:
        # Quadratic form with matrix A: <x, A x>.
        # It's gradient is (A + A.T) x, it's Hessian is (A + A.T)
        # It's Riemannian Hessian by vector product is
        #     t3f.project(t3f.matmul(A + t3f.transpose(A), vector), x)
        f = lambda x: t3f.quadratic_form(A, x, x)
        res = t3f.hessian_vector_product(f, x, vector)

    Args:
        func: function that takes TensorTrain object as input and outputs a number.
        x: point at which to compute the Hessian and on which tangent space to
          project the gradient.
      vector: `TensorTrain` object which to multiply be the Hessian.
        name: string, name of the Op.

    Returns:
        `TensorTrain`, projection of the gradient df/dx onto the tangent space at
        point x.

    See also:
        t3f.gradients
    """
  all_cores = list(x.tt_cores) + list(vector.tt_cores)
  with tf.name_scope(name, values=all_cores):
    left = decompositions.orthogonalize_tt_cores(x)
    right = decompositions.orthogonalize_tt_cores(left, left_to_right=False)
    vector_projected = riemannian.project(vector, x)
    vector_projected = shapes.expand_batch_dim(vector_projected)
    vector_projected.projection_on = x

    def new_f(new_x):
      grad = _gradients(func, x, new_x, left, right)
      grad = shapes.expand_batch_dim(grad)
      # TODO: durty hack.
      grad.projection_on = x
      return riemannian.pairwise_flat_inner_projected(grad, vector_projected)[0, 0]

    return gradients(new_f, x)


def _block_diag_hessian_vector_product(func, x, vector):
  # TODO:
  left = decompositions.orthogonalize_tt_cores(x)
  right = decompositions.orthogonalize_tt_cores(left, left_to_right=False)
  deltas = [right.tt_cores[0]] + [tf.zeros_like(cc) for cc in right.tt_cores[1:]]
  x_projection = deltas_to_tangent_space(deltas, x, left, right)
  grad = _gradients(func, x, x_projection, left, right)
  vector_projected = riemannian.project(vector, x)
  vector_deltas = tangent_space_to_deltas(vector_projected)
  grad_deltas = tangent_space_to_deltas(grad)
  final_deltas = []
  for i in range(x.ndims()):
    h = tf.reduce_sum(grad_deltas[i] * vector_deltas[i])
    cores_grad = tf.gradients(h, x_projection.tt_cores[i])[0]
    if x.is_tt_matrix():
      r1, n, m, r2 = left.tt_cores[i].shape.as_list()
    else:
      r1, n, r2 = left.tt_cores[i].shape.as_list()
    q = tf.reshape(left.tt_cores[i], (-1, r2))
    if x.is_tt_matrix():
      if i == 0:
        curr_grad = cores_grad[:, :, :, :r2]
      elif i == w.ndims() - 1:
        curr_grad = cores_grad[r1:, :, :, :]
      else:
        curr_grad = cores_grad[r1:, :, :, :r2]
    else:
      if i == 0:
        curr_grad = cores_grad[:, :, :r2]
      elif i == x.ndims() - 1:
        curr_grad = cores_grad[r1:, :, :]
      else:
        curr_grad = cores_grad[r1:, :, :r2]
    if i < x.ndims() - 1:
      proj = (tf.eye(r1 * n) - q @ tf.transpose(q))
      # TODO: multiply faster.
      delta = tf.matmul(proj, tf.reshape(curr_grad, (-1, r2)))
      delta = tf.reshape(delta, left.tt_cores[i].shape)
    else:
      delta = curr_grad
    final_deltas.append(delta)
  return deltas_to_tangent_space(final_deltas, x, left, right)
