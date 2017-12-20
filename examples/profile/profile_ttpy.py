import timeit
import numpy as np
import tt
from tt.riemannian import riemannian

matrix = tt.matrix(tt.rand(100, 10, 10))
vec = tt.matrix(tt.rand(10, 10, 10), n=[10] * 10, m=[1] * 10)
vec_100 = tt.matrix(tt.rand(10, 10, 100), n=[10] * 10, m=[1] * 10)
tens = np.random.randn(10, 10, 10, 10)
globals = {'matrix': matrix, 'vec': vec, 'vec_100': vec_100, 'tens': tens,
           'tt': tt, 'riemannian': riemannian}

logs = {'lib': 'ttpy'}

matvec_time = timeit.timeit("matrix * vec", globals=globals, number=1000) / 1000
print('Multiplying a matrix by a vector takes %f seconds.' % matvec_time)
logs['matvec_time'] = matvec_time

matmul_time = timeit.timeit("matrix * matrix", globals=globals, number=1000) / 1000
print('Multiplying %a matrix by a matrix takes %f seconds.' % (matmul_time))
logs['matmul_time'] = matmul_time

norm_time = timeit.timeit("tt.tensor.norm(matrix.tt)", globals=globals,
                          number=1000) / 1000
print('Computing the norm of a matrix takes %f seconds.' % norm_time)
logs['norm_time'] = norm_time

# Time for a Gram matrix of size 100 x 100.
gram_time = 100 * 100 * timeit.timeit("tt.dot(vec.tt, vec.tt)", globals=globals,
                          number=1000) / 1000
print('Computing the Gram matrix of size 100 x 100 takes %f seconds.' % gram_time)
logs['gram_time'] = gram_time

# Time for a Gram matrix x A x of size 100 x 100.
mat_gram_time = 100 * 100 * timeit.timeit("tt.dot(vec.tt, (matrix * vec).tt)",
                                      globals=globals, number=100) / 100
print('Computing the Gram matrix x A x of size 100 x 100 takes %f seconds.' % mat_gram_time)
logs['mat_gram_time'] = mat_gram_time


tt_svd_time = timeit.timeit("tt.vector(tens)", globals=globals,
                            number=1000) / 1000
print('TT-SVD for tensor of shape %s takes %f seconds.' % (tens.shape,
                                                           tt_svd_time))
logs['tt_svd_time'] = tt_svd_time

project_time = timeit.timeit("riemannian.project(vec.tt, vec_100.tt)", globals=globals,
                             number=1000) / 1000
print('Projecting a vector of rank 100 on a vector of rank 10 takes %f seconds.' % project_time)
logs['project_time'] = project_time

round_time = timeit.timeit("tt.tensor.round(vec_100.tt, eps=0.0, rmax=10)", globals=globals,
                             number=1000) / 1000
print('Rounding a vector of rank 100 to rank 10 takes %f seconds.' % round_time)
logs['round_time'] = round_time
