import numpy
from scipy import sparse
from scipy.sparse import linalg
from generateA import generateA
from generateDiaPrec import generateDiaPrec
from generateVecs import p_extSoln, RHS

Lx = Ly = 1.0
Nx = 100
Ny = 100
dx = Lx / Nx
dy = Ly / Ny

n = 1.0


x = numpy.linspace(dx/2.0, Lx-dx/2.0, Nx)
y = numpy.linspace(dy/2.0, Ly-dy/2.0, Ny)
X, Y = numpy.meshgrid(x, y)


A = generateA(Nx, Ny, dx, dy)
M = generateDiaPrec(A)
p = numpy.zeros(Nx * Ny)
f = RHS(X, Y, n)
p_ext = p_extSoln(X, Y, n)


print(type(A))
print(type(M))

print("A:\n", A.toarray(), "\n")
print("M:\n", M.toarray(), "\n")
print("p0:\n", p, "\n")
print("f:\n", f ,"\n")
print("Factor:\n", f/p_ext ,"\n")

#p, info = linalg.cg(A, f, p, tol=1e-15, maxiter=1000000, M=M)
p, info = linalg.bicgstab(A, f, p, tol=1e-12, M=M)
#p, info = linalg.gmres(A, f, p, tol=1e-15, restart=1)

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

