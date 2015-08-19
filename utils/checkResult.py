import argparse
import numpy
from generateVecs import p_extSoln


parser = argparse.ArgumentParser(description="Check result")

parser.add_argument("-f", "--resultFile", action="store",
        metavar="file", type=str, required=True,
        help="A text file containning result vector")

parser.add_argument("--Nx", action="store", 
        metavar="Nx", type=int, required=True, help="Nx")

parser.add_argument("--Ny", action="store", 
        metavar="Ny", type=int, required=True, help="Ny")

parser.add_argument("-n", "--n", action="store",
        metavar="n", type=float, required=False, default=1,
        help="Wavenumber which will be used to generate exact solution")

args = parser.parse_args()

result = numpy.loadtxt(args.resultFile)

if result.size != args.Nx * args.Ny:
    print("Size error:")
    print("\tThe size of the vector in the input file: ", result.size)
    print("\tThe Nx specified in the command-line parameter: ", args.Nx)
    print("\tThe Ny specified in the command-line parameter: ", args.Ny)
    print("\tThe size calculated by specified Nx and Ny: ", args.Nx*args.Ny)
    raise ValueError("Nx * Ny not equal to the size of result vector!")


Nx = args.Nx
Ny = args.Ny
n = args.n
Lx = Ly = 1.0
dx = Lx / Nx
dy = Ly / Ny


x = numpy.linspace(dx/2.0, Lx-dx/2.0, Nx)
y = numpy.linspace(dy/2.0, Ly-dy/2.0, Ny)
X, Y = numpy.meshgrid(x, y)


p_ext = p_extSoln(X, Y, n)
err = result - p_ext

print("L2-norm of error: ", numpy.linalg.norm(err, ord=None))
print("Infinity-norm of error: ", numpy.linalg.norm(err, ord=numpy.inf))


