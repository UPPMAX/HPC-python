Interactive work on the compute nodes
=====================================

.. admonition:: Learning objectives

   - Understand what an interactive session is
   - Understand why one may need an interactive session
   - How to work with an interactive session (single + multiple cores)
   - Run an interactive-friendly Python script
   - Run an interactive-unfriendly Python script
   - Gfx Launcher and On-demand desktop       

.. questions:: 
   
   - Imagine you are developing a Python script in a line-by-line fashion. How to do so best?
       - Why not do so on the login node?
       - Why not do so by using ``sbatch``?
   - What is the drawback of using an interactive node?


.. admonition:: Compute allocations in this workshop 

   - Pelle: ``uppmax2025-2-393``
   - Kebnekaise: ``hpc2n2025-151``
   - Cosmos: ``lu2025-7-106``
   - Alvis: ``naiss2025-22-934``
   - Tetralith: ``naiss2025-22-934``  
   - Dardel: ``naiss2025-22-934``

.. admonition:: Storage space for this workshop 

   - Pelle: ``/proj/hpc-python-uppmax``
   - Kebnekaise: ``/proj/nobackup/fall-courses``
   - Cosmos: ``/lunarc/nobackup/projects/lu2025-17-52``
   - Alvis: ``/mimer/NOBACKUP/groups/courses-fall-2025``
   - Tetralith: ``/proj/courses-fall-2025``
   - Dardel: ``/cfs/klemming/projects/supr/courses-fall-2005``

Introduction
------------

Some users develop Python code in a line-by-line fashion. 

- These users typically want to run a (calculation-heavy) script frequently, to test if the code works.
- However, scheduling each new line is too slow, as it can take minutes (or sometimes hours) before the new code is run through the batch system.
- Instead, there is a way to directly work with such code: use an interactive session.

Some other users want to run programs that (1) use a lot of CPU and memory, and (2) need to be persistent/available.
One good example is Jupyter. 

- Running such a program on a login nodes would harm all other users on the login node.
- Running such a program on a computer node using ``sbatch`` would not allow a user to connect to it.
- In such a case: use an interactive session.

.. admonition:: About Jupyter

   - For HPC2N, using Jupyter on HPC2N is possible, and can be done either from the OpenOnDemand portal (easy) or through a batch job and ThinLinc (harder). 
   - For UPPMAX, using Jupyter is easy. 
   - For LUNARC, using Jupyter (https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#jupyter-lab) works best using the LUNARC HPC Desktop. Go to the Applications menu, hover over Applications - Python, and select Jupyter Lab from the menu that pops up to the right.
   - For NSC, using Jupyter is easiest done through ThinLinc, but can also be used via an SSH tunnel. 
   - For PDC, you can use Jupyter easiest through the GfxLauncher in ThinLinc: https://support.pdc.kth.se/doc/login/interactive_hpc/
   - For C3SE, you can easiest use Jupyter through the OpenOnDemand portal.   

.. admonition:: In this session we will talk about 

   - interactive/salloc/srun
   - Running a Python script interactively 
   - Open-on-demand desktop 

An interactive session is a session with direct access to a compute node. Or alternatively: an interactive session is a session, in which there is no queue before a command is run on a compute node.

You can find information about the cluster hardware in the Common section: https://uppmax.github.io/HPC-python/common/understanding_clusters.html

Any longer, resource-intensive, or parallel jobs must be run through a **batch script** or an **interactive session**. 

   - Demanding work (CPU or Memory intensive) should be done on the compute nodes.
   - If you need live interaction you should start an "interactive session"
   - On Cosmos (LUNARC), Dardel (PDC), Alvis (C3SE), and Kebnekaise (HPC2N) it can be done graphically with the Desktop-On-Demand tool ``GfxLauncher`` or portal.
   - Otherwise the terminal approach will work in all centers.

The different way HPC2N, UPPMAX, LUNARC, NSC, PDC, and C3SE provide for an interactive session
-----------------------------------------------------------------------------------

Here we define an interactive session as a session with direct access to a compute node.
Or alternatively: an interactive session is a session, in which there is no queue before a command is run on a compute node.

Some centers only offer command line interactive sessions, and some also have ways of providing graphicsal interactive sessions. 

The way this differs between the centers (command line):

- HPC2N: the user remains on a login node. 
  All commands can be sent directly to the compute node using ``srun``
- UPPMAX: the user is actually on a computer node.
  Whatever command is done, it is run on the compute node
- LUNARC: the user is actually on a computer node if the correct menu option is chosen. Whatever command is done, it is run on the compute node
- NSC: the user is actually on a computer node if the correct menu option is chosen. Whatever command is done, it is run on the compute node  
- PDC: the user remains on a login node and can submit jobs to the compute node with ``srun`` *or* (recommended) the user login to the compute node with ssh after the job is allocated. Any commands are then run directly on the compute node. 
- C3SE: the user runs a shell on a compute node. 

Start an interactive session 
----------------------------

To start an interactive session, one needs to allocate resources on the cluster first.

The command to request an interactive node differs per HPC cluster:

+--------------------+-----------------+-------------+----------+---------------------------------+
| Cluster            | ``interactive`` | ``salloc``  | ``srun`` | GfxLauncher/OpenOnDemand portal |
+====================+=================+=============+==========+=================================+
| Tetralith (NSC)    | Recommended     | N/A         | N/A      | N/A                             | 
+--------------------+-----------------+-------------+----------+---------------------------------+
| Dardel (PDC)       | N/A             | Recommended | N/A      | Possible (GfxLauncher)          | 
+--------------------+-----------------+-------------+----------+---------------------------------+
| Alvis (C3SE)       | N/A             | N/A         | Works    | Recommended (OOD)               | 
+--------------------+-----------------+-------------+----------+---------------------------------+
| Kebnekaise (HPC2N) | N/A             | Recommended | N/A      | Recommended (OOD)               | 
+--------------------+-----------------+-------------+----------+---------------------------------+
| Pelle (UPPMAX)     | Recommended     | Works       | N/A      | N/A                             |
+--------------------+-----------------+-------------+----------+---------------------------------+
| Cosmos (LUNARC)    | Works           | N/A         | N/A      | Recommended (GfxLauncher)       | 
+--------------------+-----------------+-------------+----------+---------------------------------+


Start an interactive session in the simplest way (command line) 
################################################

To start an interactive session in the simplest way, as shown here:

.. tabs::

   .. tab:: Tetralith 

      The command ``interactive`` is recommended at NSC. 

      Use:

      .. code-block:: console

         interactive -A [project_name] -t HHH:MM:SS 

      Where ``[project_name]`` is the NAISS project name,
      for example ``interactive -A naiss2025-22-934``.

      If you need more CPUs/GPUs, etc. you need to ask for that as well. The default which gives 1 CPU. 

      The output will look similar to this:

      .. code-block:: console

         [x_birbr@tetralith3 ~]$ interactive -A naiss2025-22-403
         salloc: Pending job allocation 44252533
         salloc: job 44252533 queued and waiting for resources
         salloc: job 44252533 has been allocated resources
         salloc: Granted job allocation 44252533
         salloc: Waiting for resource configuration
         salloc: Nodes n340 are ready for job
         [x_birbr@n340 ~]$ 

      Note that the prompt has changed to show that one is on an interactive node.
      
   .. tab:: Dardel

      The command ``salloc`` (or OpenOnDemand through Gfx launcher) is recommended at PDC. 

      Remember that Dardel requires you to provide the **partition** as well.  
   
      .. code-block:: console
          
         salloc -A [project_name] -t HHH:MM:SS -p main

      Where ``[project_name]`` is the NAISS project name,
      for example ``salloc -A naiss2025-22-934 -t 00:10:00 -p main``.

      This will look similar to this (including asking for resources - time is required):

      .. code-block:: console

          bbrydsoe@login1:~> salloc --time=00:10:00 -A naiss2025-22-934 -p main
          salloc: Pending job allocation 9722449
          salloc: job 9722449 queued and waiting for resources
          salloc: job 9722449 has been allocated resources
          salloc: Granted job allocation 9722449
          salloc: Waiting for resource configuration
          salloc: Nodes nid001134 are ready for job
          bbrydsoe@login1:~>

      Again, you are on the login node, and anything you want to run in the allocation must be preface with srun.

      However, you have another option; you can ssh to the allocated compute node and then it will be true interactivity:

      .. code-block:: console 

         bbrydsoe@login1:~> ssh nid001134
         bbrydsoe@nid001134:~

      It is also possible to use OpenOnDemand through Gfx launcher.

      To do this, login with ThinLinc and start the Gfxlauncher application. There is some documentation here: <a href="https://support.pdc.kth.se/doc/login/interactive_hpc/" target="_blank">Interactive HPC at PDC</a>.

      Please be aware that the number of ThinLinc licenses are limited. 

      We will look more at OpenOnDemand/GfxLauncher in a short while. 

   .. tab:: Alvis 

      The command ``srun`` from command line works at C3SE. It is not recommended as when the login node is restarted the interactive job is also terminated.

      .. code-block:: console 

         [brydso@alvis2 ~]$ srun --account=NAISS2025-22-395 --gpus-per-node=T4:1 --time=01:00:00 --pty=/bin/bash
        [brydso@alvis2-12 ~]$

      The recommended way to do interactive jobs at Alvis is with OpenOnDemand.

      You access the Open OnDemand service through https://alvis.c3se.chalmers.se.

      NOTE that you need to connect from a network on SUNET.
 
      More information about C3SE’s Open OnDemand service can be found here: https://www.c3se.chalmers.se/documentation/connecting/ondemand/.   

   .. tab:: Kebnekaise

      The command ``salloc`` (or OpenOnDemand) is recommended at HPC2N.

      Usage: ``salloc -A [project_name] -t HHH:MM:SS``

      You have to give project ID and walltime. If you need more CPUs (1 is default) or GPUs, you have to ask for that as well.

      .. code-block:: console 

         b-an01 [~]$ salloc -A hpc2n2025-151 -t 00:10:00
         salloc: Pending job allocation 34624444
         salloc: job 34624444 queued and waiting for resources
         salloc: job 34624444 has been allocated resources
         salloc: Granted job allocation 34624444
         salloc: Nodes b-cn1403 are ready for job
         b-an01 [~]$

      WARNING! This is not true interactivity! Note that we are still on the login node!

      In order to run anything in the allocation, you need to preface with ``srun`` like this:

      .. code-block:: console 

          b-an01 [~]$ srun /bin/hostname
          b-cn1403.hpc2n.umu.se
          b-an01 [~]$

      Otherwise anything will run on the login node! Also, interactive sessions (for instance a program that asks for input) will not work correctly as that dialogoue happens on the compute node which you do not have real access to!

      **OpenOnDemand**

      This is the recommended way to do interactive jobs at HPC2N.

      - Go to https://portal.hpc2n.umu.se/ and login.
      - Documentation here: https://docs.hpc2n.umu.se/tutorials/connections/#open__ondemand

      More about OpenOnDemand desktop in a short while. 

   .. tab:: Pelle  

      At UPPMAX, ``interactive`` is recommended.

      Usage: ``interactive -A [project_name] -t HHH:MM:SS``

      If you need more CPUs/GPUs, etc. you need to ask for that as well. The default which gives 1 CPU.

      .. code-block:: console 

         [bbrydsoe@pelle1 ~]$ interactive -A uppmax2025-2-393 -t 00:15:00
         This is a temporary version of interactive-script for Pelle
         Most interactive-script functionality is removed
         salloc: Pending job allocation 205612
         salloc: job 205612 queued and waiting for resources
         salloc: job 205612 has been allocated resources
         salloc: Granted job allocation 205612
         salloc: Waiting for resource configuration
         salloc: Nodes p115 are ready for job
         [bbrydsoe@p115 ~]$ 

       **``salloc`` also works** 

       Usage: ``salloc -A [project_name] -t HHH:MM:SS``

       You have to give project ID and walltime. If you need more CPUs (1 is default) or GPUs, you have to ask for that as well.

       .. code-block:: console 

          [bbrydsoe@pelle1 ~]$ salloc -A uppmax2025-2-393 -t 00:15:00
          salloc: Pending job allocation 205613
          salloc: job 205613 queued and waiting for resources
          salloc: job 205613 has been allocated resources
          salloc: Granted job allocation 205613
          salloc: Nodes p115 are ready for job
          [bbrydsoe@p115 ~]$ 
       
   .. tab:: Cosmos

      The command ``interactive`` works at LUNARC. It is not the recommended way to do interactive work. 

      Usage: ``interactive -A [project_name] -t HHH:MM:SS``

      If you need more CPUs/GPUs, etc. you need to ask for that as well. The default which gives 1 CPU.

      .. code-block:: console 

         [bbrydsoe@cosmos2 ~]$ interactive -A lu2025-7-76 -t 00:15:00
         Cluster name: COSMOS
         Waiting for JOBID 1724396 to start

      After a short wait, you get something like this:

      .. code-block::  console 

         [bbrydsoe@cn094 ~]$

      **GfxLauncher**

      This is the recommended wait to work interactively at LUNARC.

      - Login with ThinLinc: https://lunarc-documentation.readthedocs.io/en/latest/getting_started/using_hpc_desktop/
      - Follow the documentation for starting the GfxLauncher for OpenOnDemand: https://lunarc-documentation.readthedocs.io/en/latest/getting_started/gfxlauncher/

      More about GfxLauncher and OpenOnDemand in a short while! 


Indeed, all you need at most of the centers, for command line interactivity, is the project name, as well as time - and also partition for PDC. 

However, this simplest way may have some default settings that do not fit you. 

- session duration is too short
- the session has too few cores available (default is usually 1) 
- or if you need GPUs 

You can add more resources the same way as for batch jobs. 

There is some information here: <https://uppmax.github.io/R-python-julia-matlab-HPC/python/interactivePython.html#start-an-interactive-session-in-a-more-elaborate-way>.

End an interactive session
--------------------------

You leave interactive mode with ``exit``. 


Check to be in an interactive session
-------------------------------------

.. admonition:: For UPPMAX, LUNARC, C3SE, and NSC (and in some cases PDC) 

   You check if you are in an interactive session with: 

   .. code-block:: console

      hostname

   If the output contains the words ``pelle``, ``cosmos``, ``tetralith``, ``alvis`` or ``login`` you are on the login node. 

   If the output contains: 

   - ``p[number].uppmax.uu.se``, where ``[number]`` is a number, you are on a compute node at UPPMAX (pelle).
   - ``cn[number]``, where ``[number]`` is a number, you are on a compute node at LUNARC (cosmos). 
   - ``n[number]``, where ``[number]`` is a number, you are on a compute node at NSC (tetralith). 
   - ``nid[number]``, where ``[number]`` is a number, you are on a compute node at PDC (dardel).  
   - ``alvis[number]-[other-number]``, where ``[number]`` is the login node you logged into (usually 2 or 3) and ``othernumber`` is a number, you are on a compute node at C3SE (alvis). 

.. admonition:: For HPC2N (and sometimes PDC) 

   You check if you are in an interactive session with: 

   .. code-block:: console

      srun hostname

   - If the output is ``b-cn[number].hpc2n.umu.se``, where ``[number]`` is a number, you are more-or-less on a compute node at Kebnekaise.

   - If the output is ``b-an[number]``, where ``[number]`` is a number, you are still on a login node on Kebnekaise.

   Do NOT do 

   .. code-block:: console

      hostname

   for HPC2N as it will always show that you are on a login node

Check that the number of cores booked is correct
------------------------------------------------

You can do this with 

.. code-block:: 

   $ srun hostname

And then you will get one line of output per core booked. 


Running a Python script in an interactive session
-------------------------------------------------

.. tabs::

   .. tab:: UPPMAX/LUNARC/C3SE/NSC/PDC (when SSH'ed to a compute node) 

      To run a Python script in an interactive session, first load the Python modules:

      .. code-block:: console

         module load [python/version + any prerequisites]

      Suggested versions (and prerequisites): 

      - UPPMAX/pelle: Python/3.12.3-GCCcore-13.3.0
      - LUNARC/cosmos: GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11
      - C3SE/Alvis: Python/3.11.3-GCCcore-12.3.0 OpenMPI/4.1.5-GCC-12.3.0 SciPy-bundle/2023.07-gfbf-2023a
      - NSC/tetralith: buildtool-easybuild/4.9.4-hpc71cbb0050 GCC/13.2.0 Python/3.11.5 
      - PDC/dardel: cray-python/3.11.7 

      To run a Python script on 1 core, do:

      .. code-block:: console

         python [my_script.py]

      where `[my_script.py]` is the Python script (including the path if it is ot in the current directory), for example ``srun python ~/my_script.py``.

      To run a Python script on each of the requested cores, do:

      .. code-block:: console

         srun python [my_script.py]

      where `[my_script.py]` is the Python script (including the path if it is noth in the current directory), for example ``srun python ~/my_script.py``.
      
   .. tab:: HPC2N

      To run a Python script in an interactive session, first load the Python modules + prerequisites:

      .. code-block:: console

         module load GCC/12.3.0 Python/3.11.3

      To run a Python script on each of the requested cores, do:

      .. code-block:: console

         srun python [my_script.py]

      where `[my_script.py]` is the Python script (including the path if it is noth in the current directory), for example ``srun python ~/my_script.py``.

Not all Python scripts are suitable for an interactive session.
This will be demonstrated by two Python example scripts.

Our first example Python script is called `sum-2args.py <https://raw.githubusercontent.com/UPPMAX/HPC-python/refs/heads/main/Exercises/examples/programs/sum-2args.py>`_:
it is a simple script that adds two numbers from command-line arguments:
 
.. code-block:: python

    import sys
  
    x = int(sys.argv[1])
    y = int(sys.argv[2])
  
    sum = x + y
  
    print("The sum of the two numbers is: {0}".format(sum))

Our second example Python script is called `add2.py <https://raw.githubusercontent.com/UPPMAX/HPC-python/refs/heads/main/Exercises/examples/programs/add2.py>`_:
it is a simple script that adds two numbers from user input:
 
.. code-block:: python

    # This program will add two numbers that are provided by the user

    # Get the numbers
    a = int(input("Enter the first number: ")) 
    b = int(input("Enter the second number: "))

    # Add the two numbers together
    sum = a + b

    # Output the sum
    print("The sum of {0} and {1} is {2}".format(a, b, sum))

.. challenge:: 

   - Why is/is it not a good script for interactive?

   - Try start an interactive session and run the scripts. 

Conclusion
----------

.. keypoints::

   You have:

   - learned a little about login nodes and compute nodes
   - seen how to use a compute node interactively,
     which differs between HPC2N, UPPMAX, LUNARC, C3SE, NSC, and PDC (particularly between HPC2N (and PDC) and the others) 
   - checked if we are in an interactive session
   - checked if we have booked the right number of cores
   - run Python scripts in an interactive session,
     which differs between HPC2N and the others
   - seen that not all Python scripts 
     can be run interactively on multiples cores
   - exited an interactive session


