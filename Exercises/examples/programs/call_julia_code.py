from time import perf_counter
import julia
from julia import Main

Main.include('julia_function.jl')

# grid size
n = 10000

if __name__ == "__main__":

    starttime = perf_counter()
    integral = Main.integration2d_julia(n)
    endtime = perf_counter()

print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
print("Time spent: %.2f sec" % (endtime-starttime))
