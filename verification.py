import numpy
from scipy import sparse
from scipy.sparse import linalg

L = 1.0
N = 1000
dL = L / N

n = 1.0

def p_extSoln(X, Y, n):
    return numpy.cos(2 * n * numpy.pi * X) * \
            numpy.cos(2 * n * numpy.pi * Y)

def RHS(X, Y, n, dL):
    coef = 8 * n * n * numpy.pi * numpy.pi * dL * dL
    return coef * numpy.cos(2 * n * numpy.pi * X) * \
            numpy.cos(2 * n * numpy.pi * Y)

x = numpy.linspace(dL/2.0, L-dL/2.0, N)
y = numpy.linspace(dL/2.0, L-dL/2.0, N)
X, Y = numpy.meshgrid(x, y)


p = numpy.zeros(N * N)
f = RHS(X, Y, n, dL).reshape(N * N)

row = numpy.empty(0)
col = numpy.empty(0)
data = numpy.empty(0)

for gN in range(N*N):

    gI = gN % N
    gJ = int(gN / N)

    if gI == 0 and gJ == 0:
        row = numpy.append(row, numpy.array([gN]))
        col = numpy.append(col, numpy.array([gN]))
        data = numpy.append(data, numpy.array([1]))
        f[gN] = p_extSoln(X[gJ, gI], Y[gJ, gI], n)
        print(gN, gI, gJ, f[gN])
    elif gI == N-1 and gJ == 0:
        row = numpy.append(row, numpy.array(3*[gN]))
        col = numpy.append(col, numpy.array([gN-1, gN, gN+N]))
        data = numpy.append(data, numpy.array([-1, 2, -1]))
    elif gI == 0 and gJ == N-1:
        row = numpy.append(row, numpy.array(3*[gN]))
        col = numpy.append(col, numpy.array([gN-N, gN, gN+1]))
        data = numpy.append(data, numpy.array([-1, 2, -1]))
    elif gI == N-1 and gJ == N-1:
        row = numpy.append(row, numpy.array(3*[gN]))
        col = numpy.append(col, numpy.array([gN-N, gN-1, gN]))
        data = numpy.append(data, numpy.array([-1, -1, 2]))
    elif gI == 0:
        row = numpy.append(row, numpy.array(4*[gN]))
        col = numpy.append(col, numpy.array([gN-N, gN, gN+1, gN+N]))
        data = numpy.append(data, numpy.array([-1, 3, -1, -1]))
    elif gI == N-1:
        row = numpy.append(row, numpy.array(4*[gN]))
        col = numpy.append(col, numpy.array([gN-N, gN-1, gN, gN+N]))
        data = numpy.append(data, numpy.array([-1, -1, 3, -1]))
    elif gJ == 0:
        row = numpy.append(row, numpy.array(4*[gN]))
        col = numpy.append(col, numpy.array([gN-1, gN, gN+1, gN+N]))
        data = numpy.append(data, numpy.array([-1, 3, -1, -1]))
    elif gJ == N-1:
        row = numpy.append(row, numpy.array(4*[gN]))
        col = numpy.append(col, numpy.array([gN-N, gN-1, gN, gN+1]))
        data = numpy.append(data, numpy.array([-1, -1, 3, -1]))
    else:
        row = numpy.append(row, numpy.array(5*[gN]))
        col = numpy.append(col, numpy.array([gN-N, gN-1, gN, gN+1, gN+N]))
        data = numpy.append(data, numpy.array([-1, -1, 4, -1, -1]))

A = sparse.csr_matrix((data, (row, col)), shape=(N*N, N*N), dtype=numpy.float)

del row, col, data

print("A:\n", A.toarray(), "\n")
print("p0:\n", p, "\n")
print("f:\n", f ,"\n")

#p = linalg.bicgstab(A, f, p, tol=1e-15)
p, info = linalg.gmres(A, f, p)
p_ext = p_extSoln(X, Y, n).reshape(N * N)

print("GMRES Info: ", info, "\n")
print("p:\n", p, "\n")
print("p exact:\n", p_ext, "\n")

