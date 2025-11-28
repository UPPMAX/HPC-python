import multiprocessing
from multiprocessing import Array
import math
import sys
from time import perf_counter

# grid size
n = 10000
# number of processes
numprocesses = 4
# partial sum for each thread
partial_integrals = Array('d',[0]*numprocesses, lock=False)

def integration2d_multiprocessing(n,numprocesses,processindex):
    global partial_integrals;
    # interval size (same for X and Y)
    h = math.pi / float(n)
    # cummulative variable
    mysum = 0.0
    # workload for each process
    workload = n/numprocesses

    begin = int(workload*processindex)
    end = int(workload*(processindex+1))
    # regular integration in the X axis
    for i in range(begin,end):
        x = h * (i + 0.5)
        # regular integration in the Y axis
        for j in range(n):
            y = h * (j + 0.5)
            mysum += math.sin(x + y)

    partial_integrals[processindex] = h**2 * mysum


if __name__ == "__main__":

    starttime = perf_counter()

    processes = []
    for i in range(numprocesses):
        p = multiprocessing.Process(target=integration2d_multiprocessing, args=(n,numprocesses,i))
        processes.append(p)
        p.start()

    # waiting for the processes
    for p in processes:
        p.join()

    integral = sum(partial_integrals)
    endtime = perf_counter()

print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
print("Time spent: %.2f sec" % (endtime-starttime))
