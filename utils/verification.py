import numpy
from scipy import sparse
from scipy.sparse import linalg

L = 1.0
N = 100
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
        row = numpy.append(row, numpy.array(3*[gN]))
        col = numpy.append(col, numpy.array([gN, gN+1, gN+N]))
        data = numpy.append(data, numpy.array([3, -1, -1]))
        f[gN] += p_extSoln(X[gJ, gI], Y[gJ, gI], n)
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

row = numpy.arange(N*N)
col = numpy.arange(N*N)
data = 1.0 / A.diagonal()
M = sparse.csr_matrix((data, (row, col)), shape=(N*N, N*N), dtype=numpy.float)
del row, col, data

print(type(A))
print(type(M))

p_ext = p_extSoln(X, Y, n).reshape(N * N)


print("A:\n", A.toarray(), "\n")
print("M:\n", M.toarray(), "\n")
print("p0:\n", p, "\n")
print("f:\n", f ,"\n")
print("Factor:\n", f/p_ext ,"\n")

p, info = linalg.cg(A, f, p, tol=1e-15, maxiter=1000000)
#p, info = linalg.cg(A, f, p, tol=1e-2, maxiter=1000000, M=M)
#p, info = linalg.bicgstab(A, f, p, tol=1e-15, M=M)
#p, info = linalg.gmres(A, f, p, tol=1e-15, restart=1)

p -= numpy.average(p)
print("p:\n", p, "\n")
print("p exact:\n", p_ext, "\n")
print(p.shape, p_ext.shape)

err = abs(p - p_ext)
L2norm = numpy.linalg.norm(err, 2)
LInfnorm = numpy.linalg.norm(err, numpy.inf)

print("err:\n", err, "\n")
print("Info: \t\t", info, "\n")
print("L2Norm:\t\t", L2norm, "\n")
print("LInfNorm:\t", LInfnorm, "\n")
