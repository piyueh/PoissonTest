import numpy
from scipy import sparse

def generateDiaPrec(A):
    '''
    '''

    N = A.shape[0]

    row = numpy.arange(N)
    col = numpy.arange(N)
    data = 1.0 / A.diagonal()

    M = sparse.csr_matrix((data, (row, col)), shape=(N, N), dtype=numpy.float)

    return M
