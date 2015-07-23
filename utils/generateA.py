import numpy
from scipy import sparse

def generateA(Nx, Ny):
    '''
    '''

    row = numpy.empty(0)
    col = numpy.empty(0)
    data = numpy.empty(0)

    N = Nx * Ny
    
    for gN in range(N):

        gI = gN % Nx
        gJ = int(gN / Nx)

        if gI == 0 and gJ == 0:
            row = numpy.append(row, numpy.array(3*[gN]))
            col = numpy.append(col, numpy.array([gN, gN+1, gN+Nx]))
            data = numpy.append(data, numpy.array([-3, 1, 1]))
        elif gI == Nx-1 and gJ == 0:
            row = numpy.append(row, numpy.array(3*[gN]))
            col = numpy.append(col, numpy.array([gN-1, gN, gN+Nx]))
            data = numpy.append(data, numpy.array([1, -2, 1]))
        elif gI == 0 and gJ == Ny-1:
            row = numpy.append(row, numpy.array(3*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN, gN+1]))
            data = numpy.append(data, numpy.array([1, -2, 1]))
        elif gI == Nx-1 and gJ == Ny-1:
            row = numpy.append(row, numpy.array(3*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN]))
            data = numpy.append(data, numpy.array([1, 1, -2]))
        elif gI == 0:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN, gN+1, gN+Nx]))
            data = numpy.append(data, numpy.array([1, -3, 1, 1]))
        elif gI == Nx-1:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN, gN+Nx]))
            data = numpy.append(data, numpy.array([1, 1, -3, 1]))
        elif gJ == 0:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-1, gN, gN+1, gN+Nx]))
            data = numpy.append(data, numpy.array([1, -3, 1, 1]))
        elif gJ == Ny-1:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN, gN+1]))
            data = numpy.append(data, numpy.array([1, 1, -3, 1]))
        else:
            row = numpy.append(row, numpy.array(5*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN, gN+1, gN+Nx]))
            data = numpy.append(data, numpy.array([1, 1, -4, 1, 1]))

    A = sparse.csr_matrix((data, (row, col)), shape=(N, N), dtype=numpy.float)

    return A
