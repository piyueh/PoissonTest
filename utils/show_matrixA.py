import numpy
import scipy
from scipy import sparse
from scipy import io

A = io.mmread("matrixA.mtx")

print("Type of Matrix A:")
print(type(A), "\n")

print("Matrix A:")
print(A.toarray(), "\n")

print("Transpose Matrix of A:")
print(A.transpose().toarray(), "\n")
