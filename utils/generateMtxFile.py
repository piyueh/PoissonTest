import argparse

import numpy
from scipy import sparse
from scipy.sparse import linalg
from scipy import io

from generateA import generateA
from generateVecs import p_extSoln, RHS



parser = argparse.ArgumentParser(
        description="Generate a MatrixMarket file for a linear problem, " +
                    "which has an exact solution" +
                    "of f(x, y) = cos(2*pi*n)*cos(2*pi*n*y)")

parser.add_argument("--Nx", action="store", 
        metavar="Nx", type=int, required=True, help="Nx")

parser.add_argument("--Ny", action="store", 
        metavar="Ny", type=int, required=True, help="Ny")

parser.add_argument("-n", "--n", action="store",
        metavar="n", type=float, required=False, default=1, help="Wavenumber")

parser.add_argument("-f", "--file", action="store",
        metavar="name", type=str, required=False, 
        default="AmgXMtxSystem.mtx", help="Output File")

args = parser.parse_args()

Lx = Ly = 1.0
Nx = args.Nx
Ny = args.Ny
dx = Lx / Nx
dy = Ly / Ny

n = args.n


x = numpy.linspace(dx/2.0, Lx-dx/2.0, Nx)
y = numpy.linspace(dy/2.0, Ly-dy/2.0, Ny)
X, Y = numpy.meshgrid(x, y)


A = generateA(Nx, Ny, dx, dy)
p = numpy.zeros(Nx * Ny)
f = RHS(X, Y, n)

io.mmwrite(args.file, A, "%AMGX rhs solution")

file = open(args.file, "ab")

numpy.savetxt(file, f, '%.18e')
numpy.savetxt(file, p, '%.18e')

file.close()

