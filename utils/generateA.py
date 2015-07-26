import numpy
from scipy import sparse

def generateA(Nx, Ny, dx, dy):
    '''
    '''

    coef_x = 1.0 / (dx**2)
    coef_y = 1.0 / (dy**2)
    coef_diag = - 2 * (coef_x + coef_y)

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
            data = numpy.append(data, 
                    numpy.array([coef_diag+coef_x+coef_y+1.0, coef_x, coef_y]))
        elif gI == Nx-1 and gJ == 0:
            row = numpy.append(row, numpy.array(3*[gN]))
            col = numpy.append(col, numpy.array([gN-1, gN, gN+Nx]))
            data = numpy.append(data, 
                    numpy.array([coef_x, coef_diag+coef_x+coef_y , coef_y]))
        elif gI == 0 and gJ == Ny-1:
            row = numpy.append(row, numpy.array(3*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN, gN+1]))
            data = numpy.append(data, 
                    numpy.array([coef_y, coef_diag+coef_x+coef_y , coef_x]))
        elif gI == Nx-1 and gJ == Ny-1:
            row = numpy.append(row, numpy.array(3*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN]))
            data = numpy.append(data, 
                    numpy.array([coef_y, coef_x, coef_diag+coef_x+coef_y ]))
        elif gI == 0:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN, gN+1, gN+Nx]))
            data = numpy.append(data, 
                    numpy.array([coef_y, coef_diag+coef_x, coef_x, coef_y]))
        elif gI == Nx-1:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN, gN+Nx]))
            data = numpy.append(data, 
                    numpy.array([coef_y, coef_x, coef_diag+coef_x, coef_y]))
        elif gJ == 0:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-1, gN, gN+1, gN+Nx]))
            data = numpy.append(data, 
                    numpy.array([coef_x, coef_diag+coef_y, coef_x, coef_y]))
        elif gJ == Ny-1:
            row = numpy.append(row, numpy.array(4*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN, gN+1]))
            data = numpy.append(data, 
                    numpy.array([coef_y, coef_x, coef_diag+coef_y, coef_x]))
        else:
            row = numpy.append(row, numpy.array(5*[gN]))
            col = numpy.append(col, numpy.array([gN-Nx, gN-1, gN, gN+1, gN+Nx]))
            data = numpy.append(data, 
                    numpy.array([coef_y, coef_x, coef_diag, coef_x, coef_y]))

    A = sparse.csr_matrix((data, (row, col)), shape=(N, N), dtype=numpy.float)

    return A
