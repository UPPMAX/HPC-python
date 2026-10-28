from mpi4py import MPI
import math
import sys
from time import perf_counter

# MPI communicator
comm = MPI.COMM_WORLD
# MPI size of communicator
numprocs = comm.Get_size()
# MPI rank of each process
myrank = comm.Get_rank()

# grid size
n = 10000

def integration2d_mpi(n,numprocs,myrank):
    # interval size (same for X and Y)
    h = math.pi / float(n)
    # cummulative variable
    mysum = 0.0
    # workload for each process
    workload = n/numprocs

    begin = int(workload*myrank)
    end = int(workload*(myrank+1))
    # regular integration in the X axis
    for i in range(begin,end):
        x = h * (i + 0.5)
        # regular integration in the Y axis
        for j in range(n):
            y = h * (j + 0.5)
            mysum += math.sin(x + y)

    partial_integrals = h**2 * mysum
    return partial_integrals


if __name__ == "__main__":

    starttime = perf_counter()

    p = integration2d_mpi(n,numprocs,myrank)

    # MPI reduction
    integral = comm.reduce(p, op=MPI.SUM, root=0)

    endtime = perf_counter()

if myrank == 0:
    print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
    print("Time spent: %.2f sec" % (endtime-starttime))
