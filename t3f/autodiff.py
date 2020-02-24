import tensorflow as tf

from t3f import shapes
from t3f import decompositions
from t3f import riemannian
from t3f import utils


def value_and_grad(f, x):
  """Gradient of the given function w.r.t. x. Works in eager and graph mode."""
  if utils.in_eager_mode():
    with tf.GradientTape() as tape:
      tape.watch(x)
      v = f(x)
    return v, tape.gradient(v, x)
  else:
    v = f(x)
    return v, tf.gradients(v, x)


def _enforce_gauge_conditions(deltas, left):
  """Project deltas that define tangent space vec onto the gauge conditions."""
  proj_deltas = []
  tt_ranks = shapes.lazy_tt_ranks(left)
  for i in range(left.ndims()):
    right_r = tt_ranks[i + 1]
    q = tf.reshape(left.tt_cores[i], (-1, right_r))
    if i < left.ndims() - 1:
      proj_delta = deltas[i]
      proj_delta = tf.reshape(proj_delta, (-1, right_r))
      proj_delta -= tf.matmul(q, tf.matmul(tf.transpose(q), proj_delta))
      proj_delta = tf.reshape(proj_delta, left.tt_cores[i].shape)
    else:
      proj_delta = deltas[i]
    proj_deltas.append(proj_delta)
  return proj_deltas


def _is_invariant_to_input_transforms(f_value_1, f_value_2,
                                      name="check_autodiff_arguments"):
  """Returns an assert op that checks that the f_value_1 == f_value_2.

  Args:
    f_value_1: tf.Tensor, value of the function computed on x_1
    f_value_2: tf.Tensor, value of the function computed on x_2
    name: String, the name of the returned op

  Here we assume that as tensors x_1 == x_2, but their TT-cores are different,
  e.g. x_2 is a cores orthogonalization version of x_1.

  The function prints a warning about introducing overhead and returns an Assert
  op that checks that the two values are reasonably close to each other.

  Returns:
    tf.op, assertion operation.
  """
  print('Warning: runtime_check of Riemannian autodiff is turned on which '
        'makes things a bit slower. It is advisable to keep runtime_check=True '
        'untill actuall production usage, since runtime check does help to '
        'catch bugs.')
  rel_diff = tf.abs((f_value_1 - f_value_2) / f_value_1)
  err_msg = "The function passed to Riemannian autodiff returns different " \
            "values for two different versions of the same tensor. " \
            "The function values are"
  assert_op = tf.Assert(rel_diff < 1e-5, [err_msg, f_value_1, f_value_2],
                        name=name)
  return assert_op


def gradients(func, x, name='t3f_gradients', runtime_check=True):
  """Riemannian autodiff: returns gradient projected on tangent space of TT.

  Computes projection of the gradient df/dx onto the tangent space of TT tensor
  at point x.

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
      runtime_check: [True] whether to do a sanity check that the passed
        function is invariant to different TT representations (otherwise
        the Rieamnnian gradient doesn't even exist). It makes things slower,
        but helps catching bugs, so turn it off during production deployment.

  Returns:
      `TensorTrain`, projection of the gradient df/dx onto the tangent space at
      point x.

  See also:
      t3f.hessian_vector_product
  """
  with tf.name_scope(name):
    left = decompositions.orthogonalize_tt_cores(x)
    right = decompositions.orthogonalize_tt_cores(left, left_to_right=False)
    deltas = [right.tt_cores[0]]
    deltas += [tf.zeros_like(cc) for cc in right.tt_cores[1:]]

    def augmented_func(d):
      x_projection = riemannian.deltas_to_tangent_space(d, x, left, right)
      return func(x_projection)

    function_value, cores_grad = value_and_grad(augmented_func, deltas)
    if runtime_check:
      assert_op = _is_invariant_to_input_transforms(function_value, func(x))
    else:
      assert_op = tf.no_op()
    with tf.control_dependencies([assert_op]):
      deltas = _enforce_gauge_conditions(cores_grad, left)
    return riemannian.deltas_to_tangent_space(deltas, x, left, right)


def hessian_vector_product(func, x, vector, name='t3f_hessian_vector_product',
                           runtime_check=True):
  """P_x [d^2f/dx^2] P_x vector, i.e. Riemannian hessian by vector product.

    Computes
      P_x [d^2f/dx^2] P_x vector
    where P_x is projection onto the tangent space of TT at point x and
    d^2f/dx^2 is the Hessian of the function.

    Note that the true Riemannian hessian also includes the manifold curvature
    term which is ignored here.

    Warning: this is experimental feature and it may not work for some function,
    e.g. ones that include QR or SVD decomposition (t3f.project, t3f.round) or
    for functions that work with TT-cores directly (in contrast to working with
    TT-object only via t3f functions). In this cases this function can silently
    return wrong results!

    Example:
        # Quadratic form with matrix A: <x, A x>.
        # It's gradient is (A + A.T) x, it's Hessian is (A + A.T)
        # It's Riemannian Hessian by vector product is
        #     proj_vec = t3f.project(vector, x)
        #     t3f.project(t3f.matmul(A + t3f.transpose(A), proj_vec), x)
        f = lambda x: t3f.bilinear_form(A, x, x)
        res = t3f.hessian_vector_product(f, x, vector)

    Args:
        func: function that takes TensorTrain object as input and outputs a number.
        x: point at which to compute the Hessian and on which tangent space to
          project the gradient.
      vector: `TensorTrain` object which to multiply be the Hessian.
      name: string, name of the Op.
      runtime_check: [True] whether to do a sanity check that the passed
        function is invariant to different TT representations (otherwise
        the Rieamnnian gradient doesn't even exist). It makes things slower,
        but helps catching bugs, so turn it off during production deployment.

    Returns:
        `TensorTrain`, result of the Riemannian hessian by vector product.

    See also:
        t3f.gradients
    """
  all_cores = list(x.tt_cores) + list(vector.tt_cores)
  with tf.name_scope(name):
    left = decompositions.orthogonalize_tt_cores(x)
    right = decompositions.orthogonalize_tt_cores(left, left_to_right=False)
    deltas = [right.tt_cores[0]]
    deltas += [tf.zeros_like(cc) for cc in right.tt_cores[1:]]

    def augmented_outer_func(deltas_outer):

      def augmented_inner_func(deltas_inner):
        x_projection = riemannian.deltas_to_tangent_space(deltas_inner, x, left,
                                                          right)
        return func(x_projection)

      function_value, cores_grad = value_and_grad(augmented_inner_func, deltas_outer)
      if runtime_check:
        assert_op = _is_invariant_to_input_transforms(function_value, func(x))
      else:
        assert_op = tf.no_op()
      with tf.control_dependencies([assert_op]):
        vector_projected = riemannian.project(vector, x)
      vec_deltas = riemannian.tangent_space_to_deltas(vector_projected)
      products = [tf.reduce_sum(a * b) for a, b in zip(cores_grad, vec_deltas)]
      return tf.add_n(products)

    _, second_cores_grad = value_and_grad(augmented_outer_func, deltas)
    final_deltas = _enforce_gauge_conditions(second_cores_grad, left)
    return riemannian.deltas_to_tangent_space(final_deltas, x, left, right)