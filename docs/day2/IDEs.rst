Loading IDEs
============

There are several popular IDEs that are commonly used for interactive work with Python. Here we will show how to load ``Jupyter``, ``VS Code``, and ``Spyder``. 

Jupyter
-------

Jupyter is web application that (among other things) allows literature programming for Python. That is, Jupyter allows to create documents where Python code is shown and run and its results shown, surrounded by written text (e.g. English).

Additionally, Jupyter allows to share files and hence includes a file manager.

Jupyter is:

- started and run on a server, for example, an interactive node

- displayed in a web browser, such as firefox.

Jupyter can be slow when using a remote desktop website (e.g. ``rackham-gui.uppmax.uu.se`` or ``kebnekaise-tl.hpc2n.umu.se``).

- For HPC2N, as JupyterLab it is only accessible from within HPC2N’s domain, and there is no way to improve any slowness

- For UPPMAX, one can use a locally installed ThinLinc client to speed up Jupyter. See the UPPMAX `documentation on ThinLinc <https://docs.uppmax.uu.se/software/thinlinc/>`_ on how to install the ThinLinc client locally

- For LUNARC, you can run Jupyter either in compute nodes through Anaconda or through the LUNARC HPC desktop. The latter is recommended. There is information about `Jupyter at LUNARC in their documentation <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#jupyter-lab>`_. 

- For NSC, you can start Thinlinc and run Jupyter on a login node, or use a browser on your local computer with SSH tunneling which could be faster. 

.. tabs::

   .. tab:: UPPMAX

      Depending on your requirement of GPU or CPU, you can use Jupyter on either Rackham or Snowy compute node.

      **1. Login to a remote desktop**

         Alt1.  Login to the remote desktop website at ``rackham-gui.uppmax.uu.se``
         Alt2. Login to your local ThinLinc client

      **2. start an interactive session**

         Start a terminal. Within that terminal, start an interactive session from the login node (change to the correct NAISS project ID)

         .. tabs::

            .. tab:: Rackham
            
               .. code-block:: sh

                  $ interactive -A uppmax2025-2-393 -t 4:00:00


            .. tab:: Snowy

               .. code-block:: sh

                  $ interactive -M snowy -A uppmax2025-2-393 -t 4:00:00 --gres=gpu:1




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

            .. tab:: Rackham

               * start ``firefox`` on the ThinLinc.
               * browse to the URLs, which will be similar to ``http://r[xxx]:8888/?token=5c3aeee9fbfc7a11c4a64b2b549622231388241c2``
               * Paste the url and it will start the Jupyter interface on ThinLinc and all calculations and files will be on Rackham.

            .. tab:: Snowy

               * start ``firefox`` on the ThinLinc.
               * browse to the URLs, which will be similar to ``http://s[xxx].uppmax.uu.se:8889/tree?token=2ac454a7c5d7376e965ad521d324595ce3d4``
               * Paste the url and it will start the Jupyter interface on ThinLinc and all calculations and files will be on Snowy.

      *On own computer*

         If you want to connect to the Jupyter server running on Rackham/Snowy from your own computer, you can do this by using SSH tunneling. Which means forwarding the port of the interactive node to your local computer.

         .. tabs::

            .. tab:: Rackham

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



            .. tab:: Snowy

               * Similar steps as for Rackham but with the correct port number and hostname pointing to Snowy compute node instead.
               
               .. code-block:: sh

                  $ ssh -L 8889:s123:8889 username@rackham.uppmax.uu.se
               
               * On your computer open the URL you got from step 3. on your webbrowser but replace s123 with localhost i.e. you get something like this

               ``http://localhost:8889/tree?token=2ac454a7c5d7376e965ad521d324595ce3d4``
               or
               ``http://127.0.0.0:8889/tree?token=2ac454a7c5d7376e965ad521d324595ce3d4``

               * Paste the url and it will start the Jupyter interface on your computer and all calculations and files will be on Snowy.



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


More information
################

- You can also check the lesson about `Jupyter on compute nodes <https://uppmax.github.io/R-python-julia-matlab-HPC/python/jupyter.html>`_ in our **Introduction to running R, Python and Julia in HPC workshop**)
- Documentation about `Jupyter on HPC2N <https://docs.hpc2n.umu.se/tutorials/jupyter/>`_
- Documentation about `Jupyter on UPPMAX <http://docs.uppmax.uu.se/software/jupyter/>`_


VS Code
--------

VS Code is a powerful and flexible IDE that is popular among developers for its ease of use and flexibility. It is designed to be a lightweight and fast editor that can be customized to suit the user's needs. It has a built-in terminal, debugger, and Git integration, and can be extended with a wide range of plugins.

VS Code can be downloaded and installed on your local machine from the `VS Code website <https://code.visualstudio.com/>`_. It is also available on the HPC center resources, but the installation process is different for each center.

VS Code is available on ThinLinc on UPPMAX and LUNARC only. On HPC2N and NSC, you will have to install it on your own laptop. 
At UPPMAX(Rackham) load it using ``module load VSCodium``, this is an open source version of VS Code. At LUNARC(Cosmos) you can find it under Applications->Programming->Visual Studio Code.

However, VS Code is best used on your local machine, as it is a resource-intensive application that can slow down the ThinLinc interface. The VS Code Server can be installed on all the HPCs that give your the ability to run your code on the HPCs but edit it on your local machine.
Similarly, you can also install your faviroute extensions on the HPCs and use them on your local machine. Care should be taken while assigning the correct installation directories for the extensions because otherwise they get installed in home directory and eat up all the space.

On your own computer through SSH tunneling 
############################################

Install VS Code on your local machine and follow the steps below to connect to the HPC center resources.

.. admonition:: Steps to connect VS Code via SSH
   :class: dropdown

   .. figure:: ../img/vscode_remote_tunnels_before_install.png
   
   .. figure:: ../img/vscode_add_new_remote.png
   
   Type ssh [username]@rackham.uppmax.uu.se where [username] is your UPPMAX username, for example, ssh sven@rackham.uppmax.uu.se. 
   This will change as per the HPC center you are using:  
   
   .. figure:: ../img/vscode_ssh_to_rackham.png
   
   Use the ~/.ssh/config file:  
   
   .. figure:: ../img/vscode_remote_tunnels_use_ssh_config_in_home.png
   
   Click on 'Connect':  
   
   .. figure:: ../img/vscode_connect_to_rackham.png
   
   .. figure:: ../img/vscode_connected_to_rackham.png

When you first establish the ssh connection to Rackham, your VSCode server directory .vscode-server will be created in your home folder /home/[username].
This also where VS Code will install all your extentions that can quickly fill up your home directory.

Install and manage Extensions on remote VSCode server
#####################################################

Manage Extensions
^^^^^^^^^^^^^^^^^

Go to Command Palette Ctrl+Shift+P or F1. Search for Remote-SSH: Settings and then go to Remote.SSH: Server Install Path. Add Item as remote host rackham.uppmax.uu.se and Value as project folder in which you want to install all your data and extensions ``/proj/uppmax202x-x-xx/nobackup`` (without a trailing slash /).

If you already had your vscode-server running and storing extensions in home directory. Make sure to kill the server by selecting Remote-SSH: KIll VS Code Server on Host on Command Palette and deleting the .vscode-server directory in your home folder.

Install Extensions
^^^^^^^^^^^^^^^^^^^

You can sync all your local VSCode extensions to the remote server after you are connected with VSCode server on HPC resource by searching for Remote: Install Local Extensions in 'SSH: rackham.uppmax.uu.se' in Command Palette. You can alternatively, go to Extensions tab and select each individually.

Selecting Kernels
^^^^^^^^^^^^^^^^^^^

Request allocation in either HPC compute node depending on your need, for that use interactive (or salloc) slurm command. Load the correct module on HPC resource that contains the interpreter you want on your VSCode. For example in case you need ML packages and python interpreter on Rackham/Snowy, do module load python_ML_packages. Check the file path for python interpreter by checking ``which python`` and copy this path. Go to Command Palette Ctrl+Shift+P or F1 on your local VSCode. Search for "interpreter" for python, then paste the path of your interpreter/kernel.

venv or conda environments are also visible on VSCode when you select interpreter/kernel for python or jupyter server. 

For jupyter, you need to start the server on the HPC resource first, check `Jupyter`_ section on how to do that. Copy the jupyter server URL which goes something like ``http://s193.uppmax.uu.se:8888/tree?token=xxx`` (in case of Snowy), click on **Select Kernel** on VSCode and select **Existing Jupyter Server**. Past the URL here and confirm your choice.
The application will automatically perform port forwarding to your local machine from the compute nodes over certain ports. Check the Terminal->Ports tab to see the correct url to open in your browser.
NOTE: Selecting kernels/interpreter does not work currently on HPC2N.

Spyder
------

Spyder is a powerful and flexible IDE originally developed to be the main scripting environment for scientific Anaconda users. It is designed to enable quick and easily repeatable experimentation, with automatic syntax checking, auto-complete suggestions, a runtime variable browser, and a graphics window that makes plots easy to manipulate after creation without additional code.

Spyder is available independent of Anaconda, but conda is still the recommended installer. Packages from the ``conda-forge`` source repo are still open-source, so conda is still usable on some facilities despite the recent changes in licensing. It is also possible to `build a pip environment with Spyder <https://docs.spyder-ide.org/current/installation.html#using-pip>`_, although this is only recommended for experienced Python users running on Linux operating systems.

To use Spyder on one of the HPC center resources, you must have a Thinlinc window open and logged into your choice of HPC resource. For personal use, it is relatively easy to `install as a standalone package on Windows or Mac <https://docs.spyder-ide.org/current/installation.html>`_, and there is also the option of `using Spyder online via Binder <https://mybinder.org/v2/gh/spyder-ide/binder-environments/spyder-stable?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fspyder-ide%252FSpyder-Workshop%26urlpath%3Ddesktop%252F%26branch%3Dmaster>`_.

.. tabs::

   .. tab:: LUNARC

      On COSMOS, the recommended way to use Spyder is to use the On-Demand version in the Applications menu, under ``Applications - Python``. All compatible packages should be configured to load upon launching, so you should only have to specify walltime and maybe a few extra resource settings with the GfxLauncher so that spyder will run on the compute nodes. Refer to `the Desktop On Demand documentation <https://uppmax.github.io/HPC-python/day1/ondemand-desktop.html>`_ to help you fill in GfxLauncher prompt.

      Avoid launching Spyder from the command line on the login node.

   .. tab:: HPC2N

      The only available version of Spyder on Kebnekaise is Spyder/4.1.5 for Python-3.8.2 (the latest release of Spyder available for users to install in their own environments is 6.0.2). Python 3.8.2 is associated with compatible versions of Matplotlib and Pandas, but not Seaborn or any of the ML packages to be covered later. To run the available version of Spyder, run the following commands:

      .. code-block:: console

         ml GCC/9.3.0  OpenMPI/4.0.3  Python  Spyder
         spyder3
      
      If you want a newer version with more and newer compatible Python packages, you will have to create a virtual environment.

   .. tab:: UPPMAX

      Spyder is not available centrally on Rackham. 

      - Use the conda env you created in Exercise 2 in `Use isolated environments <https://uppmax.github.io/HPC-python/day2/use_isolated_environments/#exercises>`_

      .. code-block:: console

         ml conda
         export CONDA_PKG_DIRS=/proj/hpc-python-uppmax/$USER
         export CONDA_ENVS_PATH=/proj/hpc-python-uppmax/$USER
         source activate spyder-env

      * you can install packages with pip install from inside Spyder

   .. tab:: NSC

      Spyder is not available on Tetralith. 

      - Use the conda env you created in Exercise 2 in `Use isolated environemnts <https://uppmax.github.io/HPC-python/day2/use_isolated_environments.html#exercises>`_

      .. code-block:: console

         module load Miniforge/24.7.1-2-hpc1
         export CONDA_PKG_DIRS=/proj/hpc-python-spring-naiss/$USER
         export CONDA_ENVS_PATH=/proj/hpc-python-spring-naiss/$USER
         source activate spyder-env

      * you can install packages with pip install from inside Spyder

   .. tab:: PDC

      Spyder is not available on Dardel.

      - Use the conda env you created in Exercise 2 in `Use isolated environemnts <https://uppmax.github.io/HPC-python/day2/use_isolated_environments.html#exercises>`_

      .. code-block:: console

         ml PDC/23.12
         ml miniconda3/24.7.1-0-cpeGNU-23.12
         export CONDA_ENVS_PATH="/cfs/klemming/projects/supr/courses-fall-2025/$USER/"
         export CONDA_PKG_DIRS="/cfs/klemming/projects/supr/courses-fall-2025/$USER/"
         source activate spyder-env

      * you can install packages with pip install from inside Spyder

Features
########

When you open Spyder, you should see something like the figure below. There should be a large pane on the left for code, and two smaller panes on the right. Each of the 3 panes have their own button with 3 horizontal lines (the menu button or "burger icon") in the top right, each with additional configuration options for those panes.

.. admonition:: Spyder interface
   :class: dropdown

   .. figure:: ../img/cosmos-on-demand-spyder.png


The top right pane has several useful tabs.

* **Help** displays information about commands and starts up with a message instructing you on how to use it.
* **Variable explorer** shows a table of currently defined variables, their datatypes, and their current values at runtime. It updates every time you either run a script file or run a command in the IPython console.
* **Files** shows the file tree for your current working directory
* Depending on the version , there may also be a **Plots** tab for displaying and adjusting graphics produced with, e.g. Matplotlib. 

You can move any of these tabs into separate windows by clicking the menu (burger) button and selecting "Undock". This is especially helpful for the plotting tab.

The bottom right pane shows the IPython console and a history tab. The IPython console is your Python command line, and runs in your current working directory unless changed with ``os.chdir()``. The default path is whatever directory you were in when you launched Spyder, but that can be changed in Preferences. If you run a script file that you've saved to a different directory using the green arrow icon on the menu ribbon, the IPython console will switch your working directory to the one containing that script. The history tab stores the last 500 lines of code excuted in the IPython console.

Most of the icons along the top menu bar under the verbal menu are running and debugging commands. You can hover over any of them to see what they do.


Configuring Spyder
##################

**Font and icon sizes.** If you are on Thinlinc and/or on a Linux machine, Spyder may be uncomfortably small the first time you open it. To fix this,

#. Click the icon shaped like a wrench (Preferences) or click "Tools" and then "Preferences". A popup should open with a menu down the left side, and the "General" menu option should already be selected (if not, select it now).
#. You should see a tab titled "Interface" that has options that mention "high-DPI scaling". Select "Set custom high-DPI scaling" and enter the factor by which you'd like the text and icons to be magnified (recommend a number from 1.5 to 2).
#. Click "Apply". If the popup in the next step doesn't appear immediately, then click "OK".
#. A pop-up will appear that says you need to restart to view the changes and asks if you want to restart now. Click "Yes" and wait. The terminal may flash some messages like ``QProcess: Destroyed while process ("/hpc2n/eb/software/Python/3.8.2-GCCcore-9.3.0/bin/python") is still running."``, but it should restart within a minute or so. Don't interrupt it or you'll have to start over.

The text and icons should be rescaled when it reopens, and should stay rescaled even if you close and reopen Spyder, as long as you're working in the same session.

**(Optional but recommended) Configure plots to open in separate windows.** In some versions of Spyder, there is a separate Plotting Pane that you can click and drag out of its dock so you can resize figures as needed, but if you don't see that, you will probably want to change your graphics backend. The default is usually "Inline", which is usually too small and not interactive. To change that,

#. Click the icon shaped like a wrench (Preferences) or click "Tools" and then "Preferences" to open the Preferences popup.
#. In the menu sidebar to the left, click "IPython console". The box to the right should then have 4 tabs, of which the second from the left is "Graphics" (see figure below).
#. Click "Graphics" and find the "Graphics backend" box below. In that box, next to "Backend" there will be a dropdown menu that probably says "Inline". Click the dropdown and select "Automatic".
#. Click "Apply" and then "OK" to exit.

.. admonition:: Spyder interface
   :class: dropdown

   .. figure:: ../img/cosmos-on-demand-spyder-preferences.png

Now, graphics should appear in their own popup that has menu options to edit and save the content.

Exercises
----------

.. challenge::

   * Try running a Scipy and a Pytorch example in your favorite IDE.  
   * Create a Virtual env using your faviroute package manager and install the packages.  
   * For an extra challenge: Run the same code in .ipynb format in your IDE. This requires you to install jupyter notebook in your virtual environment.
   
   .. admonition:: Solving linear system of equations and optimiztion task using Scipy
      :class: dropdown

      Install ``Scipy`` for the following example.

      .. code-block:: python
      
         import numpy as np
         from scipy.linalg import solve
         from scipy.optimize import minimize

         # Test 1: Solve a linear system of equations
         # Ax = b
         A = np.array([[3, 1], [1, 2]])
         b = np.array([9, 8])

         # Solve for x
         x = solve(A, b)
         print("Solution to the linear system Ax = b:")
         print(x)

         # Test 2: Minimize a simple quadratic function
         # f(x) = (x - 3)^2
         def quadratic_function(x):
            return (x - 3) ** 2

         # Initial guess
         x0 = [0]

         # Minimize the function
         result = minimize(quadratic_function, x0)
         print("\nOptimization result:")
         print("Minimum value of f(x):", result.fun)
         print("Value of x at minimum:", result.x)

   .. admonition:: Loading transformers model with pytorch backend and performing tokenization
      :class: dropdown

      Install ``transformers[torch]`` for the following example. Can be performed either on GPU or CPU node.

      .. code-block:: python

            from transformers import AutoTokenizer, AutoModel
            import torch

            # Check if GPU is available
            if torch.cuda.is_available():
               device = torch.device("cuda")
               print("GPU is available. Using:", torch.cuda.get_device_name(0))
            else:
               device = torch.device("cpu")
               print("GPU is not available. Using CPU.")

            # Load a pre-trained tokenizer and model
            model_name = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            # Move the model to GPU (if available)
            model = model.to(device)

            # Check where the model is loaded
            print(f"Model is loaded on: {device}")

            # Tokenize a sample text
            text = "Transformers library is amazing!"
            inputs = tokenizer(text, return_tensors="pt").to(device)  # Move inputs to GPU if available

            # Perform a forward pass to ensure everything works
            outputs = model(**inputs)

            # Detokenize the input IDs back to text
            detokenized_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            print("Detokenized text:", detokenized_text)

   Learning outcomes:
      * How to use IDE on any system
      * How to install packages in the environment 
