Starting IDEs from command line
===============================

.. admonition:: Learning objectives

   - Be abke to start IDEs from the command-line

      - Jupyter
      - VScode
      - spyder



Introduction
------------

Recommended to run on compute node from interactive session

This session from command line
- See last session how to start.

Next session from OnDemand.

The

- Jupyter
- VScode
- Spyder

- See last session how to start.

Jupyter
-------

Jupyter is web application that (among other things) allows literature programming for Python. That is, Jupyter allows to create documents where Python code is shown and run and its results shown, surrounded by written text (e.g. English).

Additionally, Jupyter allows to share files and hence includes a file manager.

Jupyter is:

- started and run on a server, for example, an interactive node

- displayed in a web browser, such as firefox.

Jupyter can be slow when using a remote desktop website (e.g. ``pelle-gui.uppmax.uu.se`` or ``kebnekaise-tl.hpc2n.umu.se``).

Local notes

.. tabs::   

   .. tab:: HPC2N
        
      - For HPC2N, as JupyterLab it is only accessible from within HPC2N’s domain, and there is no way to improve any slowness

   .. tab:: UPPMAX

        - For UPPMAX, one can use a locally installed ThinLinc client to speed up Jupyter. See the UPPMAX `documentation on ThinLinc <https://docs.uppmax.uu.se/software/thinlinc/>`_ on how to install the ThinLinc client locally

   .. tab:: LUNARC

- For LUNARC, you can run Jupyter either in compute nodes through Anaconda or through the LUNARC HPC desktop. The latter is recommended. There is information about `Jupyter at LUNARC in their documentation <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#jupyter-lab>`_. 

   .. tab:: NSC

      - For NSC, you can start Thinlinc and run Jupyter on a login node, or use a browser on your local computer with SSH tunneling which could be faster. 

.. tabs::

   .. tab:: UPPMAX

      **1. Login to a remote desktop**

         Alt1. Login to the remote desktop website at ``rackham-gui.uppmax.uu.se``
         Alt2. Login to your local ThinLinc client

      **2. start an interactive session**

         Start a terminal. Within that terminal, start an interactive session from the login node (change to the correct NAISS project ID)

         .. tabs::

            .. tab:: Pelle
            
               .. code-block:: sh

                  $ interactive -A uppmax2025-2-393 -t 4:00:00

      **3. start Jupyter in the interactive session**

         Within your terminal with the interactive session, load a modern Python module:

         .. code-block:: sh

            module load python/3.11.8

         Then, start ``jupyter-notebook`` (or ``jupyter-lab``):

         .. code-block:: sh

            jupyter-notebook --ip 0.0.0.0 --no-browser

         This will start a jupyter server session so leave this terminal open. The terminal will also display multiple URLs.

      **4. Connect to the running Jupyter server**

      *On ThinLinc*

         If you use the ThinLinc, depending on which Jupyter server (Rackham or Snowy) you want to launch on web browser


         .. tabs::

            .. tab:: Pelle (NEEDS UPDATE)

               * start ``firefox`` on the ThinLinc.
               * browse to the URLs, which will be similar to ``http://r[xxx]:8888/?token=5c3aeee9fbfc7a11c4a64b2b549622231388241c2``
               * Paste the url and it will start the Jupyter interface on ThinLinc and all calculations and files will be on Rackham.

            .. tab:: Bianca (NEEDS UPDATE)

               * start ``firefox`` on the ThinLinc.
               * browse to the URLs, which will be similar to ``http://s[xxx].uppmax.uu.se:8889/tree?token=2ac454a7c5d7376e965ad521d324595ce3d4``
               * Paste the url and it will start the Jupyter interface on ThinLinc and all calculations and files will be on Snowy.

      *On own computer*

         If you want to connect to the Jupyter server running on Rackham/Snowy from your own computer, you can do this by using SSH tunneling. Which means forwarding the port of the interactive node to your local computer.

         .. tabs::

            .. tab:: Pelle (NEEDS UPDATE)

               * On Linux or Mac this is done by running in another terminal. Make sure you have the ports changed if they are not at the default ``8888``.

               .. code-block:: sh

                  $ ssh -L 8888:r486:8888 username@rackham.uppmax.uu.se

               * If you use Windows it may be better to do this in the PowerShell instead of a WSL2 terminal.
               * If you use PuTTY - you need to change the settings in "Tunnels" accordingly (could be done for the current connection as well).


               * On your computer open the URL you got from step 3. on your webbrowser but replace r486 with localhost i.e. you get something like this

               ``http://localhost:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``
               or
               ``http://127.0.0.0:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``

               * This should bring the jupyter interface on your computer and all calculations and files will be on Rackham.

      .. warning::

         **Running Jupyter in a virtual environment**

         You could also use ``jupyter`` (``-lab`` or ``-notebook``) in a virtual environment.

         If you decide to use the --system-site-packages configuration you will get ``jupyter`` from the python modules you created your virtual environment with.
         However, you **won't find your locally installed packages** from that jupyter session. To solve this reinstall jupyter within the virtual environment by force:

         .. code-block:: console

            $ pip install -I jupyter

         and run:

         .. code-block:: console

            $ jupyter-notebook

         Be sure to start the **kernel with the virtual environment name**, like "Example", and not "Python 3 (ipykernel)".


   .. tab:: HPC2N

      Since the JupyterLab will only be accessible from within HPC2N's domain, it is by far easiest to do this from inside ThinLinc, so **this is highly recommended**. You can find information about using ThinLinc at `HPC2N's documentation <https://docs.hpc2n.umu.se/software/jupyter/>`_ 


      **1. Check JupyterLab version**

         At HPC2N, you currently need to start JupyterLab on a specific compute node. To do that you need a submit file and inside that you load the JupyterLab module and its prerequisites (and possibly other Python modules if you need them - more about that later).

         To see the currently available versions, do:

         .. code-block:: console

            $ module spider JupyterLab

         You then do:

         .. code-block:: console

            $ module spider JupyterLab/<version>

         for a specific <version> to see which prerequisites should be loaded first.

         *Example, loading JupyterLab/4.0.5*

         .. code-block:: console

            $ module load GCC/12.3.0 JupyterLab/4.0.5

      **2. Start Jupyter on the compute node**
      
         Make a submit file with the following content. You can use any text editor you like, e.g. ``nano`` or ``vim``.
         Something like the file below will work. Remember to change the project id after the course, how many cores you need, and how long you want the JupyterLab to be available:

         .. code-block:: slurm

            #!/bin/bash
            #SBATCH -A hpc2n2025-151
            # This example asks for 1 core
            #SBATCH -n 1
            # Ask for a suitable amount of time. Remember, this is the time the Jupyter notebook will be available! HHH:MM:SS.
            #SBATCH --time=05:00:00

            # Clear the environment from any previously loaded modules
            module purge > /dev/null 2>&1

            # Load the module environment suitable for the job
            module load GCC/12.3.0 JupyterLab/4.0.5

            # Start JupyterLab
            jupyter lab --no-browser --ip $(hostname)

         Where the flags used to the Jupyter command has the following meaning (you can use ``Jupyter --help`` and ``Jupyter lab --help``> to see extra options):

         - *lab*: This launches JupyterLab computational environment for Jupyter.
         - *- -no-browser*: Prevent the opening of the default url in the browser.
         - *- -ip=<IP address>*: The IP address the JupyterLab server will listen on. Default is 'localhost'. In the above example script I use ``$(hostname)`` to get the content of the environment variable for the hostname for the node I am allocated by the job.

         **Note** again that the JupyterLab is *only* accessible from within the HPC2N domain, so it is easiest to work on the ThinLinc.

         Submit the above submit file. Here I am calling it ``MyJupyterLab.sh``

         .. code-block:: console

            $ sbatch MyJupyterLab.sh

      **3. Connect to the running Jupyter server**
      
         Wait until the job gets resources allocated. Check the SLURM output file; when the job has resources allocated it will have a number of URLs inside at the bottom.

         The SLURM output file is as default named ``slurm-<job-id>.out`` where you get the ``<job-id>`` when you submit the SLURM submit file (from previous step).

         **NOTE**: Grab the URL with the *hostname* since the localhost one requires you to login to the compute node and so will not work!

         The file will look **similar** to this:

         .. admonition:: slurm-<job-id>.out
            :class: dropdown

            .. code-block:: console

               b-an03 [~]$ cat slurm-24661064.out
               [I 2024-03-09 15:35:30.595 ServerApp] Package jupyterlab took 0.0000s to import
               [I 2024-03-09 15:35:30.617 ServerApp] Package jupyter_lsp took 0.0217s to import
               [W 2024-03-09 15:35:30.617 ServerApp] A `_jupyter_server_extension_points` function was not found in jupyter_lsp. Instead, a `_jupyter_server_extension_paths` function was found and will be used for now. This function name will be deprecated in future releases of Jupyter Server.
               [I 2024-03-09 15:35:30.626 ServerApp] Package jupyter_server_terminals took 0.0087s to import
               [I 2024-03-09 15:35:30.627 ServerApp] Package notebook_shim took 0.0000s to import
               [W 2024-03-09 15:35:30.627 ServerApp] A `_jupyter_server_extension_points` function was not found in notebook_shim. Instead, a `_jupyter_server_extension_paths` function was found and will be used for now. This function name will be deprecated in future releases of Jupyter Server.
               [I 2024-03-09 15:35:30.627 ServerApp] jupyter_lsp | extension was successfully linked.
               [I 2024-03-09 15:35:30.632 ServerApp] jupyter_server_terminals | extension was successfully linked.
               [I 2024-03-09 15:35:30.637 ServerApp] jupyterlab | extension was successfully linked.
               [I 2024-03-09 15:35:30.995 ServerApp] notebook_shim | extension was successfully linked.
               [I 2024-03-09 15:35:31.020 ServerApp] notebook_shim | extension was successfully loaded.
               [I 2024-03-09 15:35:31.022 ServerApp] jupyter_lsp | extension was successfully loaded.
               [I 2024-03-09 15:35:31.023 ServerApp] jupyter_server_terminals | extension was successfully loaded.
               [I 2024-03-09 15:35:31.027 LabApp] JupyterLab extension loaded from /hpc2n/eb/software/JupyterLab/4.0.5-GCCcore-12.3.0/lib/python3.11/site-packages/jupyterlab
               [I 2024-03-09 15:35:31.027 LabApp] JupyterLab application directory is /cvmfs/ebsw.hpc2n.umu.se/amd64_ubuntu2004_skx/software/JupyterLab/4.0.5-GCCcore-12.3.0/share/jupyter/lab
               [I 2024-03-09 15:35:31.028 LabApp] Extension Manager is 'pypi'.
               [I 2024-03-09 15:35:31.029 ServerApp] jupyterlab | extension was successfully loaded.
               [I 2024-03-09 15:35:31.030 ServerApp] Serving notebooks from local directory: /pfs/stor10/users/home/b/bbrydsoe
               [I 2024-03-09 15:35:31.030 ServerApp] Jupyter Server 2.7.2 is running at:
               [I 2024-03-09 15:35:31.030 ServerApp] http://b-cn1520.hpc2n.umu.se:8888/lab?token=c45b36c6f22322c4cb1e037e046ec33da94506004aa137c1
               [I 2024-03-09 15:35:31.030 ServerApp]     http://127.0.0.1:8888/lab?token=c45b36c6f22322c4cb1e037e046ec33da94506004aa137c1
               [I 2024-03-09 15:35:31.030 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
               [C 2024-03-09 15:35:31.039 ServerApp]

               To access the server, open this file in a browser:
                  file:///pfs/stor10/users/home/b/bbrydsoe/.local/share/jupyter/runtime/jpserver-121683-open.html
               Or copy and paste one of these URLs:
                  http://b-cn1520.hpc2n.umu.se:8888/lab?token=c45b36c6f22322c4cb1e037e046ec33da94506004aa137c1
                  http://127.0.0.1:8888/lab?token=c45b36c6f22322c4cb1e037e046ec33da94506004aa137c1
               [I 2024-03-09 15:35:31.078 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server

         
         To access the server, go to ``file:///.local/share/jupyter/runtime/jpserver-<newest>-open.html`` from a browser within the ThinLinc session. <newest> is a number that you find by looking in the directory ``.local/share/jupyter/runtime/`` under your home directory.

         Or, to access the server you can copy and paste the URL from the file that is SIMILAR to this: ``http://b-cn1520.hpc2n.umu.se:8888/lab?token=c45b36c6f22322c4cb1e037e046ec33da94506004aa137c1``

         **NOTE** of course, do not copy the above, but the similar looking one from the file you get from running the batch script!!!

         .. admonition:: Webbrowser view
            :class: dropdown

            Start a webbrowser within HPC2N (ThinLinc interface). Open the html or put in the URL you grabbed, including the token:

            .. figure:: ../img/jupyterlab-start.png

            After a few moments JupyterLab starts up:

            .. figure:: ../img/jupyterlab_started.png

            You shut it down from the menu with "File" > "Shut Down"

      **For the course:**

         If you want to start a Jupyter with access to matplotlib and seaborn, for use with this course for the session on matplotlib, then do the following: 

         3.1. Start ThinLinc and login to HPC2N as described under `preparations <https://uppmax.github.io/HPC-python/preparations.html>`_ 

         3.2 Load these modules

            .. code-block:: console

               module load GCC/12.3.0 Python/3.11.3 OpenMPI/4.1.5 SciPy-bundle/2023.07 matplotlib/3.7.2 Seaborn/0.13.2 JupyterLab/4.0.5

         3.3. Make a submit file with this content 

            .. code-block:: 

               #!/bin/bash
               #SBATCH -A hpc2n2025-151
               # This example asks for 1 core
               #SBATCH -n 1
               # Ask for a suitable amount of time. Remember, this is the time the Jupyter notebook will be available! HHH:MM:SS.
               #SBATCH --time=05:00:00

               # Clear the environment from any previously loaded modules
               module purge > /dev/null 2>&1

               # Load the module environment suitable for the job
               module load GCC/12.3.0 Python/3.11.3 OpenMPI/4.1.5 SciPy-bundle/2023.07 matplotlib/3.7.2 Seaborn/0.13.2 JupyterLab/4.0.5 

               # Start JupyterLab
               jupyter lab --no-browser --ip $(hostname)

         3.4. Get the URL from the SLURM output file ``slurm-<job-id>.out``.

            It will be **SIMILAR** to this : ``http://b-cn1520.hpc2n.umu.se:8888/lab?token=c45b36c6f22322c4cb1e037e046ec33da94506004aa137c1``

         3.5. Open a browser inside ThinLinc and put in the URL similar to above. 


   .. tab:: LUNARC

      You can interactively launch Jupyter Lab and Notebook on COSMOS by following the steps as below:

         1. Click on Applications -> Applications - Python -> Jupyter Lab (CPU) or Jupyter Notebook (CPU)

            .. admonition:: Desktop view
                  :class: dropdown

                  .. figure:: ../img/lunarc_start_jupyter.png

         2. Configure your job parameters in the dialog box.

            .. admonition:: GfxLauncher view
                  :class: dropdown

                  .. figure:: ../img/lunarc_jupyter_configure_job.png

         3. Click Start, wait for the job to start and in few seconds a firefox browser will open with Jupyter Lab or Notebook session. If you close the firefox browser, you can connect to same Jupyter session again by clicking ‘Reconnect to Lab’.

            .. admonition:: GfxLauncher view
                  :class: dropdown

                  .. figure:: ../img/cosmos-on-demand-job-settings.png

   .. tab:: NSC

      **Through ThinLinc**

         1. Login with ThinLinc (https://www.nsc.liu.se/support/graphics/) 

            - Download the client matching your local computer's OS and install it.
            - Start the ThinLinc client.
            - Change the “Server” setting to ``tetralith.nsc.liu.se``.
            - Change the “Name” setting to your Tetralith username (e.g x_abcde).
            - Enter your cluster Tetralith password in the “Password” box.
            - Press the “Connect” button.
            - If you connect for the first time, you will see the “The server’s host key is not cached …” dialog. 

         2. Load a JupyterLab module

            - Open a terminal    
            - This is an example for JupyterLab 4.2.0

            .. code-block:: console

               $ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0

         3. Start JupyterLab

            - Type ``jupyter-lab`` in the terminal 
            - It will show some text, including telling you to open a url in a browser (inside ThinLinc/on Tetralith). If you just wait, it will open a browser with Jupyter.   

            - It will look similar to this:

            .. admonition:: Webbrowser view
                  :class: dropdown

                  .. figure:: ../img/jupyter-thinlinc-nsc.png


      **On your own computer through SSH tunneling**

         1. Either do a regular SSH or use ThinLinc to connect to tetralith (change to your own username): 

            ``ssh x_abcde@tetralith.nsc.liu.se``

         2. Change to your working directory

            ``cd <my-workdir>``

         3. Load a module with JupyterLab in (here JupyterLab 4.2.0) 

            .. code-block:: console

               $ module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0

         4. Start jupyter with the no-browser flag

            - ``jupyter-lab --no-browser``

            - You get something that looks like this: 

            .. admonition:: Terminal view
               :class: dropdown

               .. figure:: ../img/jupyter-no-browser-nsc.png

         Where I have marked a line with relevant info. Note that the port will change. 

         5. Open a second terminal, on your home computer. Input this: 

            - ``ssh -N -L localhost:88XX:localhost:88XX x_abcde@tetralith1.nsc.liu.se``

            where you change 88XX to the actual port you got, and the name to your username. In my example it would be: 

            - ``ssh -N -L localhost:8867:localhost:8867 x_birbr@tetralith1.nsc.liu.se``

            .. admonition:: Terminal view
                  :class: dropdown

                  .. figure:: ../img/local-ssh-to-nsc.png
            
         6. Now grab the line that is similar to the one I marked in 4. and which has the same port as you used in 5. 

            - Input that line (url with token) in a browser on your local machine. You wil get something similar to this: 

            .. admonition:: Webbrowser view
                  :class: dropdown

                  .. figure:: ../img/local-jupyter-lab.png


   .. tab:: PDC

      You can interactively launch Jupyter Lab and Notebook on Dardel by following the steps as below. Hopefully the ThinLinc licenses are sufficient!
      

         1. Click on Applications -> PDC-Jupyter -> Jupyter Lab or Jupyter Notebook

            .. admonition:: Desktop view
                  :class: dropdown

                  .. figure:: ../img/pdc_start_jupyter.png

         2. Configure your job parameters in the dialog box.

            .. admonition:: GfxLauncher view
                  :class: dropdown

                  .. figure:: ../img/pdc_jupyter_configure_job.png

         3. Click Start, wait for the job to start and in few seconds a firefox browser will open with Jupyter Lab or Notebook session. If you close the firefox browser, you can connect to same Jupyter session again by clicking ‘Reconnect to Lab’.

            .. admonition:: GfxLauncher view
                  :class: dropdown

                  .. figure:: ../img/pdc_reconnect_to_jupyter.png      





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




Exercises
---------

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
   - Tetralith: ``/proj/courses-fall-courses``
   - Dardel: ``/cfs/klemming/projects/supr/courses-fall-courses``
