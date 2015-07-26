import numpy


def p_extSoln(X, Y, n):
    '''
    '''

    N = X.size

    soln = numpy.cos(2 * n * numpy.pi * X) * \
           numpy.cos(2 * n * numpy.pi * Y)

    return soln.reshape(N)


def RHS(X, Y, n):
    '''
    '''

    N = X.size

    coef = - 8 * n * n * numpy.pi * numpy.pi

    rhs = coef * numpy.cos(2 * n * numpy.pi * X) * \
          numpy.cos(2 * n * numpy.pi * Y)

    rhs[0, 0] += p_extSoln(X[0, 0], Y[0, 0], n)

    return rhs.reshape(N)
