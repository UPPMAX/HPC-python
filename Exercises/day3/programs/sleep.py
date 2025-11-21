import sys
from time import perf_counter,sleep
import multiprocessing

# number of iterations
n = 6
# number of processes
numprocesses = 6

def sleep_serial(n):
    for i in range(n):
        sleep(1)


def sleep_threaded(n,numprocesses,processindex):
    # workload for each process
    workload = n/numprocesses
    begin = int(workload*processindex)
    end = int(workload*(processindex+1))
    for i in range(begin,end):
        sleep(1)

if __name__ == "__main__":

    starttime = perf_counter()   # Start timing serial code
    sleep_serial(n)
    endtime = perf_counter()

    print("Time spent serial: %.2f sec" % (endtime-starttime))


    starttime = perf_counter()   # Start timing parallel code
    processes = []
    for i in range(numprocesses):
        p = multiprocessing.Process(target=sleep_threaded, args=(n,numprocesses,i))
        processes.append(p)
        p.start()

    # waiting for the processes
    for p in processes:
        p.join()

    endtime = perf_counter()

    print("Time spent parallel: %.2f sec" % (endtime-starttime))

