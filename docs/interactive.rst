Interactive work on the compute nodes
=====================================

.. note::

   - It is possible to run Python directly on the login (including ThinLinc) nodes.
   - *Only* be done for shorter jobs or jobs that do not use a lot of resources, as the login nodes can otherwise become slow for all users. 
   - If you want to work **interactively** with your code or data, you should start an **interactive session** if it requires much CPU or RAM.
   - If you rather will run a script which won't use any interactive user input while running, you can instead start a batch job, see last session.
   
.. questions::

   - How to reach the compute/calculation nodes
   - How do I proceed to work interactively?
   
.. objectives:: 

   - Show how to reach the compute/calculation nodes on UPPMAX and HPC2N
   - Test some commands on the compute/calculation nodes

General
-------

In order to run interactively, you need to have compute nodes allocated to run on, and this is done through the SLURM system.  

Because you will have to wait until the nodes are allocated, and because you cannot know when this happens, this is not usually a recommended way to run Python, but it is possible. 

.. warning::

    (HPC2N) Do note that it is not *real* interactivity as you probably mean it, as you will have to run it as a Python script instead of by starting Python and giving commands inside it. The reason for this is that you are not actually logged into the compute node and only sees the output of the commands you run. 

Another option would be to use Jupyter notebooks. 

- This option is covered in the `extra reading for UPPMAX <https://uppmax.github.io/HPC-python/jupyter.html>`_. 
- It is somewhat convoluted to get to work correctly at HPC2N, but possible, using a submit file as an intermediate step. There is a tutorial on that `here <https://www.hpc2n.umu.se/resources/software/jupyter>`_ 
- We will also deal with Jupyter in the next session about parallel computing. 


Python "interactively" on the compute nodes 
-------------------------------------------

To run interactively, you need to allocate resources on the cluster first. 
You can use the command salloc to allow interactive use of resources allocated to your job. 
When the resources are allocated, you need to preface commands with ``srun`` in order to 
run on the allocated nodes instead of the login node. 
      
- First, you make a request for resources with ``interactive``/``salloc``, like this:

.. tabs::

   .. tab:: UPPMAX (interactive)

      .. code-block:: console
          
         $ interactive -n <tasks> --time=HHH:MM:SS -A naiss2023-22-1126
      
   .. tab:: HPC2N (salloc)

      .. code-block:: console
          
         $ salloc -n <tasks> --time=HHH:MM:SS -A hpc2nXXXX-YYY
         
      
- where <tasks> is the number of tasks (or cores, for default 1 task per core), time is given in hours, minutes, and seconds (maximum T168 hours), and then you give the id for your project (on UPPMAX this is **naiss2023-22-1126** for this course, on HPC2N it is **hpc2nXXXX-YYY**)

- Your request enters the job queue just like any other job, and ``interactive``/``salloc`` will tell you that it is waiting for the requested resources. When ``interactive``/``salloc`` tells you that your job has been allocated resources, you can interactively run programs on those resources with ``srun``. The commands you run with ``srun`` will then be executed on the resources your job has been allocated. **NOTE** If you do not preface with ``srun`` the command is run on the login node! 
      
- You can now run Python scripts on the allocated resources directly instead of waiting for your batch job to return a result. This is an advantage if you want to test your Python script or perhaps figure out which parameters are best.
                  

Example
#######

.. tip::
    
   **Type along!**

**Requesting 4 cores for 10 minutes, then running Python**

.. tabs::

   .. tab:: UPPMAX

      .. code-block:: console
      
          [bjornc@rackham2 ~]$ interactive -A naiss2023-22-1126 -p devcore -n 4 -t 10:00
          You receive the high interactive priority.
          There are free cores, so your job is expected to start at once.
      
          Please, use no more than 6.4 GB of RAM.
      
          Waiting for job 29556505 to start...
          Starting job now -- you waited for 1 second.
          
          [bjornc@r484 ~]$ module load python/3.9.5

      Let us check that we actually run on the compute node: 

      .. code-block:: console
      
          [bjornc@r483 ~]$ srun hostname
          r483.uppmax.uu.se
          r483.uppmax.uu.se
          r483.uppmax.uu.se
          r483.uppmax.uu.se

      We are. Notice that we got a response from all four cores we have allocated.   

   .. tab:: HPC2N
         
      .. code-block:: console
      
          $ salloc -n 4 --time=00:10:00 -A hpc2nXXXX-YYY
          salloc: Pending job allocation 20174806
          salloc: job 20174806 queued and waiting for resources
          salloc: job 20174806 has been allocated resources
          salloc: Granted job allocation 20174806
          salloc: Waiting for resource configuration
          salloc: Nodes b-cn0241 are ready for job
          b-an01 [~]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5
          b-an01 [~]$ 
                  
      
      Let us check that we actually run on the compute node: 
      
      .. code-block:: console
                  
           $ srun hostname
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se
      
      We are. Notice that we got a response from all four cores we have allocated.   
      
      
**I am going to use the following two Python codes for the examples:**
      
      Adding two numbers from user input (add2.py)
         
      .. code-block:: python
      
          # This program will add two numbers that are provided by the user
          
          # Get the numbers
          a = int(input("Enter the first number: ")) 
          b = int(input("Enter the second number: "))
          
          # Add the two numbers together
          sum = a + b
          
          # Output the sum
          print("The sum of {0} and {1} is {2}".format(a, b, sum))
      
      Adding two numbers given as arguments (sum-2args.py)
         
      .. code-block:: python
      
          import sys
          
          x = int(sys.argv[1])
          y = int(sys.argv[2])
          
          sum = x + y
          
          print("The sum of the two numbers is: {0}".format(sum))
      
**Now for running the examples:**

- Note that the commands are the same for both HPC2N and UPPMAX!
      
      1. Running a Python script in the allocation we made further up. Notice that since we asked for 4 cores, the script is run 4 times, since it is a serial script
         
      .. code-block:: console
      
          $ srun python sum-2args.py 3 4
          The sum of the two numbers is: 7
          The sum of the two numbers is: 7
          The sum of the two numbers is: 7
          The sum of the two numbers is: 7
          b-an01 [~]$             
                  
      2. Running a Python script in the above allocation, but this time a script that expects input from you.
         
      .. code-block:: console        
          
          $ srun python add2.py 
          2
          3
          Enter the first number: Enter the second number: The sum of 2 and 3 is 5
          Enter the first number: Enter the second number: The sum of 2 and 3 is 5
          Enter the first number: Enter the second number: The sum of 2 and 3 is 5
          Enter the first number: Enter the second number: The sum of 2 and 3 is 5
      
      As you can see, it is possible, but it will not show any interaction it otherwise would have. This is how it would look on the login node: 
                  
      .. code-block:: console
                  
                  $ python add2.py 
                  Enter the first number: 2
                  Enter the second number: 3
                  The sum of 2 and 3 is 5
      

**Exit**

When you have finished using the allocation, either wait for it to end, or close it with ``exit``

.. tabs::

   .. tab:: UPPMAX
   
      .. code-block:: console
                  
                  [bjornc@r484 ~]$ exit
      
                  exit
                  [screen is terminating]
                  Connection to r484 closed.
      
                  [bjornc@rackham2 ~]$

   .. tab:: HPC2N
   
      .. code-block:: console
                  
                  $ exit
                  exit
                  salloc: Relinquishing job allocation 20174806
                  salloc: Job allocation 20174806 has been revoked.
                  $

.. admonition:: Running Jupyter on compute nodes at 

   - UPPMAX: https://uppmax.github.io/HPC-python/jupyter.html
   - HPC2N: https://www.hpc2n.umu.se/resources/software/jupyter 


.. keypoints::

   - Start an interactive session on a calculation node by a SLURM allocation
   
      - At HPC2N: ``salloc`` ...
      - At UPPMAX: ``interactive`` ...
      
   - Follow the same procedure as usual by loading the Python module and possible prerequisites.
   - CPU-hours are more effectively used in "batch jobs". Therefore:
   
     - Use "interactive" for testing and developing
     - Don't book too many cores/nodes and try to be effective when the session is going.
     
    
