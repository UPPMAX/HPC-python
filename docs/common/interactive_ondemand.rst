Interactive sessions and Desktop On-Demand
##########################################

.. note::

   - Demanding work (CPU or Memory intensive) should be done on the compute nodes.
   - If you need live interaction you should satrt an "interactive session"
   - On Cosmos (LUNARC) and Dardel (PDC) it can be done graphically with the Desktop-On-Demand tool ``GfxLauncher``.
   - Otherwise the terminal approach will work in all centers.

The command to request an interactive node differs per HPC cluster:

+---------+-----------------+-------------+-------------+
| Cluster | ``interactive`` | ``salloc``  | GfxLauncher |
+=========+=================+=============+=============+
| HPC2N   | Works           | Recommended | N/A         |
+---------+-----------------+-------------+-------------+
| UPPMAX  | Recommended     | Works       | N/A         |
+---------+-----------------+-------------+-------------+
| LUNARC  | Works           | N/A         | Recommended | 
+---------+-----------------+-------------+-------------+
| NSC     | Recommended     | N/A         | N/A         | 
+---------+-----------------+-------------+-------------+ 
| PDC     | N/A             | Recommended | Possible    | 
+---------+-----------------+-------------+-------------+ 

Start an interactive session from OnDemand 
==========================================

For most programs, the start-up process is roughly the same:

#. Log into COSMOS/(Dardel) via Thinlinc to start a  HPC Desktop session.
#. Click ``Applications`` in the top left corner and hover over the items prefixed with ``Applications -`` until you find your desired application.
#. Upon clicking your chosen application, a pop-up, the Gfx Launcher interface, will appear and let you set the following options:
      #. **Wall time** - how long your interactive session will remain open. When it ends, the whole window closes immediately and any unsaved work is lost. You can select the time from a drop-down menu, or type in the time manually. CPU-only applications can run for up to 168 hours (7 days), but the rest are limited to 48 hours. Default is 30 minutes.
      #. **Requirements** - how many tasks per node you need. The default is usually 1 or 4 tasks per node. There is also a **gear icon** to the right of this box that can pull up a second menu where you can set the name of your job, the number of tasks per node, the amount of memory per CPU core, and/or toggle whether or not to use a full node.
      #. **Resource** - which kind of node you want in terms of the architecture (AMD or Intel) and number of cores in the CPU (or GPU). Options and defaults vary by program, but it is recommended that you leave the default node type in place.
      #. **Project** - choose from a drop-down menu the project with which your work is associated. This is mainly to keep your usage in line with your licenses and permissions, and to send any applicable invoices to the correct PI. Licensed software will only work for projects whose group members are covered by the license.

   .. figure:: ../img/cosmos-on-demand-resource-specs.png
      :width: 600
      :align: center

      The GfxLauncher GUI (here used to launch Spyder). The box on the left is the basic menu and the box on the right is what pops up when the gear icon next to ``Requirements`` is clicked.


4. When you're happy with your settings, click "Start". The Gfx Launcher menu will stay open in the background so that you can monitor your remaining time and resources with the ``Usage`` bar.

If you want, you can also look at the associated SLURM scripts by clicking the "More" button at the bottom of the Gfx Launcher menu and clicking the "Script" tab (example below), or view the logs under the "Logg" tab.

   .. figure:: ../img/cosmos-on-demand-more.png
      :width: 400
      :align: center

For a few applications (e.g. Jupyter Lab), GfxLauncher will also offer an additional menu item titled ``Job settings...``. This is where you can load custom environments or additional modules if absolutely necessary. However, this feature is still a work in progress; any module already in the module box when you first open ``Job settings`` is likely necessary to run the program, and searching for additional modules (Select modules button) tends to erase any listed previously. For now, additional modules must be entered by hand (not always including the version number) in a comma-separated list. Moreover, incompatible and redundant modules tend to make the application shut down as soon as it is queued, raising a spurious alert that the requested walltime has expired.

   .. figure:: ../img/cosmos-on-demand-job-settings.png
      :width: 550
      :align: center

      The Job Properties menu (right) pops up when the box titled ``Job settings...`` in the main GfxLauncher window (left) is clicked. Only use it if you know what you're doing!

Start an interactive session from the terminal
==============================================

To start an interactive session in the simplest way, is shown here:

.. tabs::

   .. tab:: UPPMAX

      Use:

      .. code-block:: console

         interactive -A [project_name]

      Where ``[project_name]`` is the UPPMAX project name,
      for example ``interactive -A uppmax2025-2-296``.

      The output will look similar to this:

      .. code-block:: console

          [richel@rackham4 ~]$ interactive -A uppmax2025-2-296
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

                        User Guides: https://docs.uppmax.uu.se/

                        Write to support@uppmax.uu.se, if you have questions or comments.


          [richel@r314 ~]$ 

      Note that the prompt has changed to show that one is on an interactive node.
      
   .. tab:: HPC2N

      .. code-block:: console
          
         salloc -A [project_name]

      Where ``[project_name]`` is the HPC2N project name,
      for example ``salloc -A hpc2n2025-076``.

      This will look similar to this (including asking for resources - time is required):

      .. code-block:: console

          b-an01 [~]$ salloc -n 4 --time=00:10:00 -A hpc2n2025-076
          salloc: Pending job allocation 20174806
          salloc: job 20174806 queued and waiting for resources
          salloc: job 20174806 has been allocated resources
          salloc: Granted job allocation 20174806
          salloc: Waiting for resource configuration
          salloc: Nodes b-cn0241 are ready for job
          b-an01 [~]$ module load GCC/12.3.0 Python/3.11.3
          b-an01 [~]$ 

   .. tab:: LUNARC 

      .. code-block:: console 

         interactive -A [project_name]

      Where ``[project_name]`` is the LUNARC project name,
      for example ``interactive -A lu2025-7-34``.  

      This will look similar to this (including asking for resources - time is required): 

      .. code-block:: console

         [bbrydsoe@cosmos3 ~]$ interactive -A lu2025-7-34 -n 4 -t 00:10:00
         Cluster name: COSMOS
         Waiting for JOBID 988025 to start

      The terminal will refresh for the new connection: 

      .. code-block:: console

         [bbrydsoe@cn137 ~]$ module load GCC/13.2.0 Python/3.11.5
         [bbrydsoe@cn137 ~]$ 

   .. tab:: NSC 

      .. code-block:: console 

         interactive -A [project_name]

      Where ``[project_name]`` is the NSC project name,
      for example ``interactive -A naiss2025-22-403``.  

      This will look similar to this: 

      .. code-block:: console

         [x_birbr@tetralith1 ~]$ interactive -A naiss2025-22-403 
         salloc: Pending job allocation 40137281
         salloc: job 40137281 queued and waiting for resources
         salloc: job 40137281 has been allocated resources
         salloc: Granted job allocation 40137281
         salloc: Waiting for resource configuration
         salloc: Nodes n302 are ready for job
         [x_birbr@n302 ~]$ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5
         [x_birbr@n302 ~]$

  .. tab:: PDC (salloc)

      .. code-block:: console 

         salloc -A [project_name]

      Where ``[project_name]`` is the PDC project name,
      for example ``interactive -A naiss2025-22-403``.  

      This will look similar to this: 

      .. code-block:: console

         claremar@login1:~> salloc --ntasks=4 -t 0:30:00 -p shared --qos=normal -A naiss2025-22-403
         salloc: Pending job allocation 9102757
         salloc: job 9102757 queued and waiting for resources
         salloc: job 9102757 has been allocated resources
         salloc: Granted job allocation 9102757
         salloc: Waiting for resource configuration
         salloc: Nodes nid001057 are ready for job

      We need to ssh to the specific node

      .. code-block:: console

         ssh nid001057

However, this simplest way may have some defaults settings that do not fit you. 

- session duration is too short
- the session has too few cores available

You can add more resources the same way as for batch jobs.

End an interactive session
--------------------------

You leave interactive mode with ``exit``. 


Check to be in an interactive session
-------------------------------------

.. admonition:: For UPPMAX, LUNARC, PDC and NSC 

   You check if you are in an interactive session with: 

   .. code-block:: console

      hostname

   If the output contains the words ``rackham``, ``cosmos``, or ``tetralith`` you are on the login node. 

   If the output contains: 

   - ``r[number].uppmax.uu.se``, where ``[number]`` is a number, you are on a compute node at UPPMAX (rackham).
   - ``cn[number]``, where ``[number]`` is a number, you are on a compute node at LUNARC (cosmos). 
   - ``n[number]``, where ``[number]`` is a number, you are on a compute node at NSC (tetralith). 

.. admonition:: For HPC2N 

   You check if you are in an interactive session with: 

   .. code-block:: console

      srun hostname

   - If the output is ``b-cn[number].hpc2n.umu.se``, where ``[number]`` is a number, you are more-or-less on a compute node.

   - If the output is ``b-an[number]``, where ``[number]`` is a number, you are still on a login node.

   Do NOT do 

   .. code-block:: console

      hostname

   for HPC2n as it will always show that you are on a login node

Check that the number of cores booked is correct
------------------------------------------------

You can do this on all clusters, except for Dardel and Cosmos, with 

.. code-block:: 

   $ srun hostname

And then you will get one line of output per core booked. 

On Dardel instead test

.. code-block:: console
                  
   claremar@nid001027:~> srun -n 4 hostname
   nid001027
   nid001027
   nid001027
   nid001027

Now, it seems that Dardel allows for "hyperthreading", that is 2 threads per core.

.. code-block:: console

   claremar@nid001027:~> srun -n 8 hostname
   nid001027
   nid001027
   nid001027
   nid001027
   nid001027
   nid001027
   nid001027
   nid001027
   claremar@nid001027:~> srun -n 9 hostname
   srun: error: Unable to create step for job 9702490: More processors requested than permitted

On Cosmos instead do:

.. code-block:: console
      
   [bjornc@cn050 ~]$ echo $SLURM_CPUS_ON_NODE
   4



