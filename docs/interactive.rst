Interactive work on the compute nodes
=====================================


.. tabs::

   .. tab:: Learning objectives

      - Understand what an interactive session is
      - Understand why one may need an interactive session
      - Start an interactive session
      - Test to be on an interactive node with the right amount of cores
      - End an interactive session
      - Start an interactive session with multiple cores
      - Test to be on an interactive node with multiple cores
      - Run an interactive-friendly Python script on multiple cores
      - Run an interactive-unfriendly Python script on multiple cores
      - End an interactive session with multiple cores

   .. tab:: For teachers

      Teaching goals are:

      - Learners have heard what an interactive session is
      - Learners have heard why one may need an interactive session
      - Learners have started an interactive session
      - Learners have tested to be on an interactive node
      - Learners have ended an interactive session
      - Learners have started an interactive session with multiple cores
      - Learners have tested to be on an interactive node with multiple cores
      - Learners have ended an interactive session with multiple cores

      Shortened lesson plan (20 minutes in total):

      - 5 mins: prior knowledge
         - What types of nodes do our HPC clusters have?
         - What is the purpose of each of these nodes?
         - Imagine you are developing a Python script in a line-by-line fashion. How to do so best?
         - Why not do so on the login node?
         - Why not do so by using ``sbatch``?
      - 5 mins: presentation
      - 5 mins: challenge
      - 5 mins: recap
         - What is the drawback of using an interactive node?


.. admonition:: Compute allocations in this workshop 

   - Rackham: ``naiss2024-22-415``
   - Kebnekaise: ``<your-HPC2N-project-id>``

.. admonition:: Storage space for this workshop 

   - Rackham: ``/proj/r-py-jl``
   - Kebnekaise: ``/proj/nobackup/<your-HPC2N-project-id>``

Introduction
------------

Some users develop Python code in a line-by-line fashion. 
These users typically want to run a (calculation-heavy) 
script frequently, to test if the code works.
However, scheduling each new line is too slow, as it
can take minutes before the new code is run.
Instead, there is a way to directly work 
with such code: use an interactive session.

Some other users want to run programs that 
(1) use a lot of CPU and memory, and (2) need to be persistent/available.
One good example is Jupyter. 
Running such a program on a login nodes would
harm all other users on the login node.
Running such a program on a computer node using ``sbatch``
would not allow a user to connect to it.
In such a case: use an interactive session.

.. admonition:: **About Jupyter**

    For HPC2N, using 
    `Jupyter on HPC2N <https://www.hpc2n.umu.se/resources/software/jupyter>`_ is possible, 
    yet harder to get to work correctly
    If you need it anyway, please contact ``support@hpc2n.umu.se``.

    For UPPMAX, using Jupyter is easier 
    and this will be shown in this course, in `the UPPMAX-only session on Jupyter <https://uppmax.github.io/HPC-python/jupyter.html>`_.

An interactive session is a session with direct access to a compute node.
Or alternatively: an interactive session is a session,
in which there is no queue before a command is run on a compute node.

In this session, we show how to:
- the different way HPC2N and UPPMAX provide for an interactive session
- start an interactive session
- check to be in an interactive session
- check to have booked the expected amount of cores
- end the interactive session

The different way HPC2N and UPPMAX provide for an interactive session
---------------------------------------------------------------------

.. mermaid:: mermaid/interactive_node_transitions.mmd 

Here we define an interactive session as a session 
with direct access to a compute node.
Or alternatively: an interactive session is a session,
in which there is no queue before a command is run on a compute node.

This differs between HPC2N and UPPMAX:

- HPC2N: the user remains on a login node. 
  All commands can be sent directly to the compute node using ``srun``
- UPPMAX: the user is actually on a computer node.
  Whatever command is done, it is run on the compute node

Start an interactive session
----------------------------

To start an interactive session, 
one needs to allocate resources on the cluster first.

The command to request an interactive node differs per HPC cluster:

+---------+-----------------+-------------+
| Cluster | ``interactive`` | ``salloc``  |
+=========+=================+=============+
| HPC2N   | Works           | Recommended |
+---------+-----------------+-------------+
| UPPMAX  | Recommended     | Works       |
+---------+-----------------+-------------+

Start an interactive session in the simplest way
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start an interactive session in the simplest way, is shown here:

.. tabs::

   .. tab:: UPPMAX

     Use:

      .. code-block:: console

         interactive -A [project_name]

      Where ``[project_name]`` is the UPPMAX project name,
      for example ``interactive -A naiss2024-22-415``.

      The output will look similar to this:

      .. code-block:: console

          [richel@rackham4 ~]$ interactive -A naiss2024-22-415
          You receive the high interactive priority.
          You may run for at most one hour.
          Your job has been put into the devcore partition and is expected to start at once.
          (Please remember, you may not simultaneously have more than one devel/devcore job, running or queued, in the batch system.)

          Please, use no more than 8 GB of RAM.

          salloc: Pending job allocation 9093699
          salloc: job 9093699 queued and waiting for resources
          salloc: job 9093699 has been allocated resources
          salloc: Granted job allocation 9093699
          salloc: Waiting for resource configuration
          salloc: Nodes r314 are ready for job
           _   _ ____  ____  __  __    _    __  __
          | | | |  _ \|  _ \|  \/  |  / \   \ \/ /   | System:    r314
          | | | | |_) | |_) | |\/| | / _ \   \  /    | User:      richel
          | |_| |  __/|  __/| |  | |/ ___ \  /  \    | 
           \___/|_|   |_|   |_|  |_/_/   \_\/_/\_\   | 

          ###############################################################################

                        User Guides: http://www.uppmax.uu.se/support/user-guides
                        FAQ: http://www.uppmax.uu.se/support/faq

                        Write to support@uppmax.uu.se, if you have questions or comments.


          [richel@r314 ~]$ 

      Note that the prompt has changed to show that one is on an interactive node.
      
   .. tab:: HPC2N

      .. code-block:: console
          
         salloc -A [project_name]

      Where ``[project_name]`` is the HPC2N project name,
      for example ``interactive -A hpc2n2024-025``.

      This will look similar to this:

      .. code-block:: console

          b-an01 [~]$ salloc -n 4 --time=00:10:00 -A hpc2n2024-025 
          salloc: Pending job allocation 20174806
          salloc: job 20174806 queued and waiting for resources
          salloc: job 20174806 has been allocated resources
          salloc: Granted job allocation 20174806
          salloc: Waiting for resource configuration
          salloc: Nodes b-cn0241 are ready for job
          b-an01 [~]$ module load GCC/12.3.0 Python/3.11.3
          b-an01 [~]$ 

Indeed, all you need is the UPPMAX/HPC2N project name.
However, this simplest way may have some defaults settings 
that do not fit you.

Start an interactive session in a more elaborate way
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to start an interactive session
may have some defaults settings that do not fit you:

- session duration is too short
- the session has too few cores available

Here we show how start an interactive session in a more elaborate way,
with a custom session duration and a custom amount of cores:
      
.. tabs::

   .. tab:: UPPMAX

      Here we start an interactive session on the ``devcore`` partition,
      with a custom session duration and a custom amount of cores:

      .. code-block:: console
          
         interactive -p devcore -n [n_tasks] --time=[duration] -A naiss2024-22-415

      where ``[n_tasks]`` is the number of tasks,
      ``[duration]`` is the time given in ``HHH:MM:SS`` format,
      and ``[project_name]`` is the UPPMAX project name.

      The parameters ``-p devcore`` mean that the ``devcore`` partition is used,
      which results in jobs that start either faster or just as fast. Nice!

      As an example, here an interactive job is started with 4 tasks,
      for 1 hour, for the UPPMAX project ``naiss2024-22-415``:

      .. code-block:: console

         interactive -p devcore -n 4 --time=1:00:00 -A naiss2024-22-415

      Note that, as Slurm uses 1 task per core by default, we request 4 cores.

      The output will be similar to this:

      .. code-block:: console
      
          [bjornc@rackham2 ~]$ interactive -A naiss2024-22-415 -p devcore -n 4 -t 10:00
          You receive the high interactive priority.
          There are free cores, so your job is expected to start at once.
      
          Please, use no more than 6.4 GB of RAM.
      
          Waiting for job 29556505 to start...
          Starting job now -- you waited for 1 second.

      
   .. tab:: HPC2N

      Here we start an interactive session,
      with a custom session duration and a custom amount of cores:

      .. code-block:: console
          
         interactive -n [n_tasks] --time=[duration] -A naiss2024-22-415

      where ``[n_tasks]`` is the number of tasks,
      ``[duration]`` is the time given in ``HHH:MM:SS`` format,
      and ``[project_name]`` is the HPC2N project name.

      As an example, here an interactive job is started with 4 tasks,
      for 1 hour, for the HPC2N project ``hpc2n2024-025``:

      .. code-block:: console
          
         salloc -n 4 --time=1:00:00 -A hpc2n2024-025

      Note that, as Slurm uses 1 task per core by default, we request 4 cores.

Check to be in an interactive session
-------------------------------------

.. tabs::

   .. tab:: UPPMAX

      To check to be in an interactive session, do:

      .. code-block:: console

         hostname

      If the output is ``r[number].uppmax.uu.se``, where ``[number]``
      is a number, you are on a computer node. Well done!

      If the output is ``rackham[number].uppmax.uu.se``, where ``[number]``
      is a number, you are still on a login node.
      
   .. tab:: HPC2N

      To check to be in an interactive session, do:

      .. code-block:: console

         srun hostname

      If the output is ``b-cn[number].hpc2n.umu.se``, where ``[number]``
      is a number, you are more-or-less on a computer node. Well done!

      If the output is ``[something else]``, where ``[number]``
      is a number, you are still on a login node.

      This is an example of output when 4 cores have been booked:

      .. code-block:: console
                  
           b-an01 [~]$ srun hostname
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se

      Misleading would be to use:

      .. code-block:: console

         hostname

      This will always show that you are on a login node

Check to have booked the expected amount of cores
-------------------------------------------------

.. tabs::

   .. tab:: UPPMAX

      To check to have booked the expected amount of cores:

      .. code-block:: console

         srun hostname

      The output should be one line of ``r[number].uppmax.uu.se``, where ``[number]``
      is a number, you have booked one core.

      If the output is more than one line of ``r[number].uppmax.uu.se``, where ``[number]``
      is a number, you have booked more than one core. 

      If the output is ``rackham[number].uppmax.uu.se``, where ``[number]``
      is a number, you are still on a login node.

      Here is an example of output when 4 cores had been booked:

      .. code-block:: console
      
          [bjornc@r483 ~]$ srun hostname
          r483.uppmax.uu.se
          r483.uppmax.uu.se
          r483.uppmax.uu.se
          r483.uppmax.uu.se
      
   .. tab:: HPC2N

      To check to have booked the expected amount of cores:

      .. code-block:: console

         srun hostname

      The output should be one line of ``b-cn[number].hpc2n.umu.se``, where ``[number]``
      is a number, you have booked one core.

      If the output is more than one line of ``b-cn[number].hpc2n.umu.se``, where ``[number]``
      is a number, you have booked more than one core. 

      If the output is ``[something else]``, where ``[number]``
      is a number, you are still on a login node.

      This is an example of output when 4 cores have been booked:

      .. code-block:: console
                  
           b-an01 [~]$ srun hostname
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se
           b-cn0241.hpc2n.umu.se

Example
#######

.. tip::
    
   **Type along!**

**Requesting 4 cores for 10 minutes, then running Python**

.. tabs::

   .. tab:: UPPMAX

      .. code-block:: console
      
          [bjornc@rackham2 ~]$ interactive -A naiss2024-22-415 -p devcore -n 4 -t 10:00
          You receive the high interactive priority.
          There are free cores, so your job is expected to start at once.
      
          Please, use no more than 6.4 GB of RAM.
      
          Waiting for job 29556505 to start...
          Starting job now -- you waited for 1 second.
          
          [bjornc@r484 ~]$ module load python/3.11.8

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
      
          $ salloc -n 4 --time=00:10:00 -A hpc2n2024-052
          salloc: Pending job allocation 20174806
          salloc: job 20174806 queued and waiting for resources
          salloc: job 20174806 has been allocated resources
          salloc: Granted job allocation 20174806
          salloc: Waiting for resource configuration
          salloc: Nodes b-cn0241 are ready for job
          b-an01 [~]$ module load GCC/12.3.0 Python/3.11.3
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
                  
                  $ exit
      
                  exit
                  [screen is terminating]
                  Connection to r484 closed.
      
                  $

   .. tab:: HPC2N
   
      .. code-block:: console
                  
                  $ exit
                  exit
                  salloc: Relinquishing job allocation 20174806
                  salloc: Job allocation 20174806 has been revoked.
                  $

.. admonition:: Running Jupyter on compute nodes at 

   - UPPMAX: https://uppmax.github.io/R-python-julia-HPC/python/jupyter.html#uppmax
   - HPC2N: https://uppmax.github.io/R-python-julia-HPC/python/jupyter.html#kebnekaise


.. keypoints::

   - Start an interactive session on a calculation node by a SLURM allocation
   
      - At HPC2N: ``salloc`` ...
      - At UPPMAX: ``interactive`` ...
      
   - Follow the same procedure as usual by loading the Python module and possible prerequisites.
   - CPU-hours are more effectively used in "batch jobs". Therefore:
   
     - Use "interactive" for testing and developing
     - Don't book too many cores/nodes and try to be effective when the session is going.
     
    
