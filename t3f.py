import numpy as np
import tensorflow as tf

def to_tt_matrix(mat, shape, max_tt_rank=10, eps=1e-6):
    """Converts a given matrix or vector to a TT-matrix.

    The matrix dimensions should factorize into d numbers.
    If e.g. the dimensions are prime numbers, it's usually better to
    pad the matrix with zeros until the dimensions factorize into
    (ideally) 3-8 numbers.

    Args:

        mat: two dimensional tf.tensor (a matrix).

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
        TT-matrix (a list of tf.tensors that you can pass to other functions
        that expect a TT-matrix as unput).

    """

    return

def to_tt_tensor(tens, max_tt_rank=10, eps=1e-6):
    """Converts a given tf.tensor to a TT-tensor of the same shape.

    Args:

        tens: tf.tensor

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
        TT-tensor (a list of tf.tensors that you can pass to other functions
        that expect a TT-tensor as unput).

    """
    return

def full_matrix(tt_matrix):
    """Converts a TT-matrix into a regular matrix (tf.tensor).

    Args:

        tt_matrix: TT-matrix to be converted.

    Returns:
        tf.tensor of rank 2.

    """
    return

def full_tensor(tt_tensor):
    """Converts a TT-tensor into a regular tf.tensor.

    Args:

        tt_tensor: TT-tensor to be converted.

    Returns:
        tf.tensor

    """
    return
