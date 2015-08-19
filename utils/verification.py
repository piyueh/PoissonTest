import argparse
import numpy
from scipy import sparse
from scipy.sparse import linalg

from generateA import generateA
from generateDiaPrec import generateDiaPrec
from generateVecs import p_extSoln, RHS

import time


parser = argparse.ArgumentParser(
        description="Solve the 2D Poisson problem, which has an exact solution" +
                    "of f(x, y) = cos(2*pi*n)*cos(2*pi*n*y)")

parser.add_argument("--Nx", action="store", 
        metavar="Nx", type=int, required=True, help="Nx")

parser.add_argument("--Ny", action="store", 
        metavar="Ny", type=int, required=True, help="Ny")

parser.add_argument("-n", "--n", action="store",
        metavar="n", type=float, required=False, default=1, help="Wavenumber")

parser.add_argument("-p", "--precd", action="store",
        metavar="bool", type=bool, required=False, default=False, 
        help="Whether to use diagnoal preconditioner or not. Default is False.")

parser.add_argument("-tol", "--tol", action="store",
        metavar="tolerance", type=float, required=False, default=1e-12, 
        help="Tolerance used in the solver. Default is 1e-12")

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
p_ext = p_extSoln(X, Y, n)

print("A:\n", A.toarray(), "\n")
if args.precd == True:
    M = generateDiaPrec(A)
    print("M:\n", M.toarray(), "\n")
print("p0:\n", p, "\n")
print("f:\n", f ,"\n")
print("Factor:\n", f/p_ext ,"\n")

bg = time.clock()
if args.precd == True:
    p, info = linalg.bicgstab(A, f, p, tol=args.tol, M=M)
else:
    p, info = linalg.bicgstab(A, f, p, tol=args.tol)
ed = time.clock()

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

print("CPU time of solve: ", ed - bg)
