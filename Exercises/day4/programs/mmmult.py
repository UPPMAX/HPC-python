import timeit
import numpy as np

starttime = timeit.default_timer()

np.random.seed(1701)

A = np.random.randint(-1000, 1000, size=(8,4))
B = np.random.randint(-1000, 1000, size =(4,4))

print("This is matrix A:\n", A)
print("The shape of matrix A is ", A.shape)
print()
print("This is matrix B:\n", B)
print("The shape of matrix B is ", B.shape)
print()
print("Doing matrix-matrix multiplication...")
print()

C = np.matmul(A, B)

print("The product of matrices A and B is:\n", C) 

print("The shape of the resulting matrix is ", C.shape)

print() 

print("Time elapsed for generating matrices and multiplying them is ", timeit.default_timer() - starttime)


