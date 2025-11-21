from time import perf_counter
import numpy as np

A = np.random.rand(3000,3000)
starttime = perf_counter()
B = np.dot(A,A)
endtime = perf_counter()

print("Time spent: %.2f sec" % (endtime-starttime))
