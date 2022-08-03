Interactive work on the compute nodes
=====================================

There are several ways to run Python interactively

- Directly on the login nodes: **only** do this for short jobs that do not take a lot of resources
- As an interactive job on the computer nodes, launched via the batch system
- Jupyter notebooks (UPPMAX suggests installing your own version with conda) 

UPPMAX
------



HPC2N
-----

It is possible to run Python directly on the Kebnekaise login node or the Kebnekaide ThinLinc login node, but this should *only* be done for shorter jobs or jobs that do not use a lot of resources, as the login nodes can otherwise become slow for all users. Both Python and IPython exists as modules to load and run. 

Another option is to either submit a batch job or to run *interactively* on the compute nodes. In order to run interactively, you need to have compute nodes allocated to run on, and this is done through the batch system.  

Because you will have to wait until the nodes are allocated, and because you cannot know when this happens, this is not a recommended way to run Python, but it is possible. 

Do note that it is not *real* interactivity as you probably mean it, as you will have to run it as a Python script instead of by starting Python and giving commands inside it. The reason for this is that you are not actually logged into the compute node and only sees the output of the commands you run. 

Another option would be to use Jupyter notebooks. This is somewhat convoluted to get to work correctly at HPC2N, but possible. Please contact us at support@hpc2n.umu.se if you want to go this route. 
            
**Python "interactively" on the compute nodes**

To run interactively, you need to allocate resources on the cluster first. You can use the command salloc to allow interactive use of resources allocated to your job. When the resources are allocated, you need to preface commands with ``srun`` in order to run on the allocated nodes instead of the login node. 

First, you make a request for resources with ``salloc``, like this:

.. code-block:: sh
    
   $ salloc -n <tasks> --time=HHH:MM:SS -A SNICXXXX-YY-ZZZ 

where <tasks> is the number of tasks (or cores, for default 1 task per core), time is given in hours, minutes, and seconds (maximum T168 hours), and then you give the id for your project (SNIC2022-22-641 for this course)
    
Your request enters the job queue just like any other job, and salloc will tell you that it is waiting for the requested resources. When salloc tells you that your job has been allocated resources, you can interactively run programs on those resources with ``srun``. The commands you run with ``srun`` will then be executed on the resources your job has been allocated. If you do not preface with ``srun`` the command is run on the login node! 

.. admonition:: Example, Requesting 4 cores for 30 minutes, then running Python 
    :class: dropdown
   
        .. code-block:: sh

            b-an01 [~]$ salloc -n 4 --time=00:30:00 -A SNIC2022-22-641
            salloc: Pending job allocation 20174806
            salloc: job 20174806 queued and waiting for resources
            salloc: job 20174806 has been allocated resources
            salloc: Granted job allocation 20174806
            salloc: Waiting for resource configuration
            salloc: Nodes b-cn0241 are ready for job
            b-an01 [~]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5
            b-an01 [~]$ 

You can now run Python scripts on the allocated resources directly instead of waiting for your batch job to return a result. This is an advantage if you want to test your Python script or perhaps figure out which parameters are best.
            
Let us check that we actually run on the compute node: 

.. code-block:: sh
            
     b-an01 [~]$ srun hostname
     b-cn0241.hpc2n.umu.se
     b-cn0241.hpc2n.umu.se
     b-cn0241.hpc2n.umu.se
     b-cn0241.hpc2n.umu.se

We are. Notice that we got a response from all four cores we have allocated.             
            
.. admonition:: Example, Running a Python script in the above allocation. Notice that since we asked for 4 cores, the script is run 4 times, since it is a serial script
    :class: dropdown
   
        .. code-block:: sh

            b-an01 [~]$ srun python sum-2args.py 3 4
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            b-an01 [~]$             
            
.. admonition:: Example, Running a Python script in the above allocation, but this time a script that expects input from you.
    :class: dropdown
   
        .. code-block:: sh            
            
            b-an01 [~]$ srun python add2.py 
            2
            3
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5

As you can see, it is possible, but it will not show any interaction it otherwise would have. This is how it would look on the login node: 
            
.. code-block:: sh 
            
            b-an01 [~]$ python add2.py 
            Enter the first number: 2
            Enter the second number: 3
            The sum of 2 and 3 is 5

When you have finished using the allocation, either wait for it to end, or close it with ``exit``
            
.. code-block:: sh 
            
            b-an01 [~]$ exit
            exit
            salloc: Relinquishing job allocation 20174806
            salloc: Job allocation 20174806 has been revoked.
            b-an01 [~]$ 
