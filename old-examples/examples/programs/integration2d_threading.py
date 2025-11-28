import threading
import math
import sys
from time import perf_counter

# grid size
n = 10000
# number of threads
numthreads = 4
# partial sum for each thread
partial_integrals = [None]*numthreads

def integration2d_threading(n,numthreads,threadindex):
    global partial_integrals;
    # interval size (same for X and Y)
    h = math.pi / float(n)
    # cummulative variable
    mysum = 0.0
    # workload for each thread
    workload = n/numthreads
    # lower and upper integration limits for each thread
    begin = int(workload*threadindex)
    end = int(workload*(threadindex+1))
    # regular integration in the X axis
    for i in range(begin,end):
        x = h * (i + 0.5)
        # regular integration in the Y axis
        for j in range(n):
            y = h * (j + 0.5)
            mysum += math.sin(x + y)

    partial_integrals[threadindex] = h**2 * mysum


if __name__ == "__main__":

    starttime = perf_counter()
    # start the threads
    threads = []
    for i in range(numthreads):
        t = threading.Thread(target=integration2d_threading, args=(n,numthreads,i))
        threads.append(t)
        t.start()

    # waiting for the threads
    for t in threads:
        t.join()

    integral = sum(partial_integrals)
    endtime = perf_counter()

print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
print("Time spent: %.2f sec" % (endtime-starttime))
