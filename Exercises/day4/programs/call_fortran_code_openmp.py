from time import perf_counter
import myfunction_openmp
import numpy

# grid size
n = 10000

if __name__ == "__main__":

    starttime = perf_counter()
    integral = myfunction_openmp.integration2d_fortran_openmp(n)
    endtime = perf_counter()

print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
print("Time spent: %.2f sec" % (endtime-starttime))
