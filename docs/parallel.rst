Parallel computing with Python
==============================

.. objectives::

   - Learn common schemes for the parallelization of codes
   - Learn general concepts for parallel computing

In Python there are different schemes that can be used to parallelize Python codes. 
We will only take a look at some of these schemes that illustrate the general concepts of
parallel computing.

The workhorse for this section will be a 2D integration example:

:math:`\int^{\pi}_{0}\int^{\pi}_{0}\sin(x+y)dxdy = 0`

One way to perform the integration is by creating a grid in the ``x`` and ``y`` directions.
More specifically, one divides the integration range in both directions into ``n`` bins. A
serial code can be seen in the following code block.

   .. admonition:: ``integration2d_serial.py``
      :class: dropdown

      .. code-block:: python

         import math
         import sys
         from time import perf_counter
         
         # grid size
         n = 10000
         
         def integration2d_serial(n):
             global integral;
             # interval size (same for X and Y)
             h = math.pi / float(n)
             # cummulative variable 
             mysum = 0.0
             
             # regular integration in the X axis
             for i in range(n):
                 x = h * (i + 0.5)
                 # regular integration in the Y axis
                 for j in range(n):
                     y = h * (j + 0.5)
                     mysum += math.sin(x + y)
             
             integral = h**2 * mysum
         
         
         if __name__ == "__main__":
         
             starttime = perf_counter()
             integration2d_serial(n)
             endtime = perf_counter()
         
         print("Integral value is %e, Error is %e" % (integral, abs(integral - 0.0)))
         print("Time spent: %.2f sec" % (endtime-starttime))


Text some text  


.. code-block:: sh 

    $ python integration2d_serial.py

Text some text  


Threads
-------

Distributed
-----------

MPI
---

Additional information
----------------------
         
* `On parallel software engineering education using python <https://link.springer.com/article/10.1007/s10639-017-9607-0>`_
