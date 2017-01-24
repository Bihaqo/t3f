import numpy as np
import tensorflow as tf

# TODO: add complexities to the comments.

def to_tt_matrix(mat, shape, max_tt_rank=10, eps=1e-6):
    """Converts a given matrix or vector to a TT-matrix.

    The matrix dimensions should factorize into d numbers.
    If e.g. the dimensions are prime numbers, it's usually better to
    pad the matrix with zeros until the dimensions factorize into
    (ideally) 3-8 numbers.

    Args:
        mat: two dimensional tf.Tensor (a matrix).
        shape: two dimensional array (np.array or list of lists)
            Represents the tensor shape of the matrix.
            E.g. for a (a1 * a2 * a3) x (b1 * b2 * b3) matrix `shape` should be
            ((a1, a2, a3), (b1, b2, b3))
            `shape[0]`` and `shape[1]`` should have the same length.
            For vectors you may use ((a1, a2, a3), (1, 1, 1)) or, equivalently,
            ((a1, a2, a3), None)
        max_tt_rank: a number or a list of numbers
            If a number, than defines the maximal TT-rank of the result.
            If a list of numbers, than `max_tt_rank` length should be d-1
            (where d is the length of `shape[0]`) and `max_tt_rank[i]` defines
            the maximal (i+1)-th TT-rank of the result.
            The following two versions are equivalent
                `max_tt_rank = r`
            and
                `max_tt_rank = r * np.ones(d-1)`
        eps: a floating point number
            If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
            the result would be guarantied to be `eps` close to `mat`
            in terms of relative Frobenious error:
                ||res - mat||_F / ||mat||_F <= eps
            If the TT-ranks are restricted, providing a loose `eps` may reduce
            the TT-ranks of the result.
            E.g.
                to_tt_matrix(mat, shape, max_tt_rank=100, eps=1)
            will probably return you a TT-matrix with TT-ranks close to 1, not 100.

    Returns:
        `TensorTrain` object containing a TT-matrix.

    """

    return

def to_tt_tensor(tens, max_tt_rank=10, eps=1e-6):
    """Converts a given tf.Tensor to a TT-tensor of the same shape.

    Args:
        tens: tf.Tensor
        max_tt_rank: a number or a list of numbers
            If a number, than defines the maximal TT-rank of the result.
            If a list of numbers, than `max_tt_rank` length should be d-1
            (where d is the rank of `tens`) and `max_tt_rank[i]` defines
            the maximal (i+1)-th TT-rank of the result.
            The following two versions are equivalent
                `max_tt_rank = r`
            and
                `max_tt_rank = r * np.ones(d-1)`
        eps: a floating point number
            If the TT-ranks are not restricted (`max_tt_rank=np.inf`), then
            the result would be guarantied to be `eps` close to `tens`
            in terms of relative Frobenious error:
                ||res - tens||_F / ||tens||_F <= eps
            If the TT-ranks are restricted, providing a loose `eps` may
            reduce the TT-ranks of the result.
            E.g.
                to_tt_tensor(tens, max_tt_rank=100, eps=1)
            will probably return you a TT-tensor with TT-ranks close to 1,
            not 100.

    Returns:
        `TensorTrain` object containing a TT-tensor.

    """
    return

def full(tt):
    """Converts a TensorTrain into a regular tensor or matrix (tf.Tensor).

    Args:
        tt: `TensorTrain` object.

    Returns:
        tf.Tensor.

    """
    return

def tt_tt_matmul(tt_matrix_a, tt_matrix_b):
    """Multiplies two TT-matrices and returns the TT-matrix of the result.

    Args:
        tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
        tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P

    Returns
        `TensorTrain` object containing a TT-matrix of size M x P
    """
    return

def tt_dense_matmul(tt_matrix_a, matrix_b):
    """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.

    Args:
        tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
        matrix_b: tf.Tensor of size N x P

    Returns
        tf.Tensor of size M x P
    """
    return

def dense_tt_matmul(matrix_a, tt_matrix_b):
    """Multiplies a regular matrix by a TT-matrix, returns a regular matrix.

    Args:
        matrix_a: tf.Tensor of size M x N
        tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P

    Returns
        tf.Tensor of size M x P
    """
    return

def sparse_tt_matmul(sparse_matrix_a, tt_matrix_b):
    """Multiplies a sparse matrix by a TT-matrix, returns a regular matrix.

    Args:
        sparse_matrix_a: tf.SparseTensor of size M x N
        tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P

    Returns
        tf.Tensor of size M x P
    """
    return

# TODO: add flag `return_type = (TT | dense)`?
def tt_sparse_matmul(tt_matrix_a, sparse_matrix_b):
    """Multiplies a TT-matrix by a sparse matrix, returns a regular matrix.

    Args:
        tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
        sparse_matrix_b: tf.SparseTensor of size N x P

    Returns
        tf.Tensor of size M x P
    """
    return

def matmul(matrix_a, matrix_b):
    """Multiplies two matrices that can be TT-, dense, or sparse.

    Note that multiplication of two TT-matrices returns a TT-matrix with much
    larger ranks.

    Args:
        matrix_a: `TensorTrain`, tf.Tensor, or tf.SparseTensor of size M x N
        matrix_b: `TensorTrain`, tf.Tensor, or tf.SparseTensor of size N x P

    Returns
        If both arguments are `TensorTrain` objects, returns a `TensorTrain`
            object containing a TT-matrix of size M x P
        If not, returns tf.Tensor of size M x P
    """
    return

def tt_tt_flat_inner(tt_a, tt_b):
    """Inner product between two TT-tensors or TT-matrices along all axis.

    The shapes of tt_a and tt_b should coincide.

    Args:
        tt_a: `TensorTrain` object
        tt_b: `TensorTrain` object

    Returns
        a number
        sum of products of all the elements of tt_a and tt_b
    """
    return

def tt_dense_flat_inner(tt_a, dense_b):
    """Inner product between a TT-tensor (or TT-matrix) and tf.Tensor along all axis.

    The shapes of tt_a and dense_b should coincide.

    Args:
        tt_a: `TensorTrain` object
        dense_b: tf.Tensor

    Returns
        a number
        sum of products of all the elements of tt_a and dense_b
    """
    return

def tt_sparse_flat_inner(tt_a, sparse_b):
    """Inner product between a TT-tensor (or TT-matrix) and tf.SparseTensor along all axis.

    The shapes of tt_a and sparse_b should coincide.

    Args:
        tt_a: `TensorTrain` object
        sparse_b: tf.SparseTensor

    Returns
        a number
        sum of products of all the elements of tt_a and sparse_b
    """
    return

def dense_tt_flat_inner(dense_a, tt_b):
    """Inner product between a tf.Tensor and TT-tensor (or TT-matrix) along all axis.

    The shapes of dense_a and tt_b should coincide.

    Args:
        dense_a: `TensorTrain` object
        tt_b: tf.SparseTensor

    Returns
        a number
        sum of products of all the elements of dense_a and tt_b
    """
    return

def sparse_tt_flat_inner(sparse_a, tt_b):
    """Inner product between a tf.SparseTensor and TT-tensor (or TT-matrix) along all axis.

    The shapes of sparse_a and tt_b should coincide.

    Args:
        sparse_a: `TensorTrain` object
        tt_b: tf.SparseTensor

    Returns
        a number
        sum of products of all the elements of sparse_a and tt_b
    """
    return

def flat_inner(a, b):
    """Inner product along all axis.

    The shapes of a and b should coincide.

    Args:
        a: `TensorTrain` tf.Tensor, or tf.SparseTensor
        b: `TensorTrain`, tf.Tensor, or tf.SparseTensor

    Returns
        a number
        sum of products of all the elements of a and b
    """
    return
