# Interactive work on the compute nodes

There are several ways to run Python interactively

- Directly on the login nodes: **only** do this for short jobs that do not take a lot of resources
- As an interactive job on the computer nodes, launched via the batch system
- Jupyter notebooks (UPPMAX suggests installing your own version with conda) 

UPPMAX
------

HPC2N
-----

It is possible to run Python directly on the Kebnekaise login node or the Kebnekaide ThinLinc login node, but this should *only* be done for short jobs or jobs that do not use a lot of resources, as the login nodes can otherwise become slow for all users. Both Python and IPython exists as modules to load and run. 

Another option is to either submit a batch job or to run *interactively* on the compute nodes. In order to run interactively, you need to have compute nodes allocated to run on, and this is done through the batch system.  

Because you will have to wait until the nodes are allocated, and because you cannot know when this happens, this is not a recommended way to run Python, but it is possible. 
            
Python interactively on the compute nodes
+++++++++++++++++++++++++++++++++++++++++

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
            salloc: Pending job allocation 20171394
            salloc: job 20171394 queued and waiting for resources
            salloc: job 20171394 has been allocated resources
            salloc: Granted job allocation 20171394
            salloc: Waiting for resource configuration
            salloc: Nodes b-cn0824 are ready for job
            b-an01 [~]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5
            b-an01 [~]$ 
