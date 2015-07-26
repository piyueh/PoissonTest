import numpy
from scipy import sparse
from scipy.sparse import linalg
from scipy import io
from generateA import generateA
from generateVecs import p_extSoln, RHS


L = 1.0
Nx = Ny = 1000
dL = L / Nx

n = 1.0


x = numpy.linspace(dL/2.0, L-dL/2.0, Nx)
y = numpy.linspace(dL/2.0, L-dL/2.0, Ny)
X, Y = numpy.meshgrid(x, y)


A = generateA(Nx, Ny, dL, dL)
p = numpy.zeros(Nx * Ny)
f = RHS(X, Y, n)

io.mmwrite("AmgXsystem.mtx", A, "%AMGX rhs solution")

file = open("AmgXsystem.mtx", "ab")

numpy.savetxt(file, f, '%.18e')
numpy.savetxt(file, p, '%.18e')

file.close()

