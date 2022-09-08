from __future__ import division
from numba import cuda, float32
import numpy
import math
from time import perf_counter

# grid size
n = 100*1024
threadsPerBlock = 16
blocksPerGrid = int((n+threadsPerBlock-1)/threadsPerBlock)

# interval size (same for X and Y)
h = math.pi / float(n)

@cuda.jit
def dotprod(C):
    # using the shared memory in the thread block
    shared = cuda.shared.array(shape=(threadsPerBlock), dtype=float32)

    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    shrIndx = cuda.threadIdx.x

    if tid >= n:
        return

    #cummulative variable
    mysum = 0.0
    # fine-grain integration in the X axis
    x = h * (tid + 0.5)
    # regular integration in the Y axis
    for j in range(n):
        y = h * (j + 0.5)
        mysum += math.sin(x + y)

    shared[shrIndx] = mysum

    cuda.syncthreads()

    # reduction for the whole thread block
    s = 1
    while s < cuda.blockDim.x:
        if shrIndx % (2*s) == 0:
            shared[shrIndx] += shared[shrIndx + s]
        s *= 2
        cuda.syncthreads()
    # collecting the reduced value in the C array
    if shrIndx == 0:
        C[cuda.blockIdx.x] = shared[0]

# array for collecting partial sums on the device
C_global_mem = cuda.device_array((blocksPerGrid),dtype=numpy.float32)

starttime = perf_counter()
dotprod[blocksPerGrid,threadsPerBlock](C_global_mem)
res = C_global_mem.copy_to_host()
integral = h**2 * sum(res)
endtime = perf_counter()

print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
print("Time spent: %.2f sec" % (endtime-starttime))

