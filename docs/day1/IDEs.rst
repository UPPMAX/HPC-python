Loading IDEs
============

There are several popular IDEs that are commonly used for interactive work with Python. Here we will show how to load ``Jupyter``, ``VScode``, and ``Spyder``. 

Jupyter
-------

Jupyter is web application that (among other things) allows literature programming for Python. That is, Jupyter allows to create documents where Python code is shown and run and its results shown, surrounded by written text (e.g. English).

Additionally, Jupyter allows to share files and hence includes a file manager.

Jupyter is:

- started and run on a server, for example, an interactive node

- displayed in a web browser, such as firefox.

Jupyter can be slow when using a remote desktop website (e.g. ``rackham-gui.uppmax.uu.se`` or ``kebnekaise-tl.hpc2n.umu.se``).

- For HPC2N, as JupyterLab it is only accessible from within HPC2N’s domain, and there is no way to improve any slowness

- For UPPMAX, one can use a locally installed ThinLinc client to speed up Jupyter. See the UPPMAX `documentation on ThinLinc on how to install the ThinLinc client locally <https://docs.uppmax.uu.se/software/thinlinc/>`_ 

- For LUNARC, you can run Jupyter either in compute nodes through Anaconda or through the LUNARC HPC desktop. The latter is recommended. There is information about `Jupyter at LUNARC in their documentation <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/#jupyter-lab>`_. 

- For NSC, you can start Thinlinc and run Jupyter on a login node, or use a browser on your local computer with SSH tunneling which could be faster. 

UPPMAX
######

UPPMAX procedure step 1: login to a remote desktop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Login to a remote desktop:

- Login to the remote desktop website at ``rackham-gui.uppmax.uu.se``
- Login to your local ThinLinc client

UPPMAX procedure step 2: start an interactive session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start a terminal. Within that terminal,
start an interactive session from the login node
(change to the correct NAISS project ID)

**For Rackham**

.. code-block:: sh

   $ interactive -A <naiss-project-id>  -t 4:00:00

**For Snowy**

.. code-block:: sh

   $ interactive -M snowy -A <naiss-project-id>  -t 4:00:00


UPPMAX procedure step 3: start Jupyter in the interactive session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Within your terminal with the interactive session,
load a modern Python module:

.. code-block:: sh

   module load python/3.11.8

Then, start ``jupyter-notebook`` (or ``jupyter-lab``):

.. code-block:: sh

   jupyter-notebook --ip 0.0.0.0 --no-browser

Leave this terminal open.

UPPMAX procedure step 4: connect to the running notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The terminal will display multiple URLs.

If you use the remote desktop website:

- start ``firefox`` on the remote desktop
- browse to the first URL, which will be similar to ``file://domus/h1/[username]/.local/share/jupyter/runtimejpserver-[number]-open.html``

In both cases, you can access Jupyter from your local computer

- start ``firefox`` on your local computer
- browse to the second URL, which will be **similar** to
  ``http://r486:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``

On own computer
'''''''''''''''

- If you use ssh to connect to Rackham, you need to forward the port of the interactive node to your local computer.
    - On Linux or Mac this is done by running in another terminal. Make sure you have the ports changed if they are not at the default ``8888``.

.. code-block:: sh

   $ ssh -L 8888:r486:8888 username@rackham.uppmax.uu.se

    - If you use Windows it may be better to do this in the PowerShell instead of a WSL2 terminal.
    - If you use PuTTY - you need to change the settings in "Tunnels" accordingly (could be done for the current connection as well).

.. figure:: ../img/putty.png
   :width: 450
   :align: center

[SSH port forwarding](https://uplogix.com/docs/local-manager-user-guide/advanced-features/ssh-port-forwarding)

On your computer open the address you got but replace r486 with localhost i.e. you get something like this

``http://localhost:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``
or
``http://127.0.0.0:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``

   This should bring the jupyter interface on your computer and all calculations and files will be on Rackham.


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



HPC2N
#####

Since the JupyterLab will only be accessible from within HPC2N's domain, it is by far easiest to do this from inside ThinLinc, so **this is highly recommended**. You can find information about using ThinLinc at `HPC2N's documentation <https://docs.hpc2n.umu.se/tutorials/jupyter/>`_ 

1. At HPC2N, you currently need to start JupyterLab on a specific compute node. To do that you need a submit file and inside that you load the JupyterLab module and its prerequisites (and possibly other Python modules if you need them - more about that later).

To see the currently available versions, do

``module spider JupyterLab``

You then do

``module spider JupyterLab/<version>``

for a specific <version> to see which prerequisites should be loaded first.

**Example, loading ``JupyterLab/4.0.5``**

``module load GCC/12.3.0 JupyterLab/4.0.5``

2. Making the submit file

Something like the file below will work. Remember to change the project id after the course, how many cores you need, and how long you want the JupyterLab to be available:

.. code-block:: slurm

   #!/bin/bash
   #SBATCH -A hpc2n2024-142
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

- **lab**: This launches JupyterLab computational environment for Jupyter.
- **--no-browser**: Prevent the opening of the default url in the browser.
- **--ip=<IP address>**: The IP address the JupyterLab server will listen on. Default is 'localhost'. In the above example script I use ``$(hostname)`` to get the content of the environment variable for the hostname for the node I am allocated by the job.

**Note** again that the JupyterLab is *only* accessible from within the HPC2N domain, so it is easiest to work on the ThinLinc.

3. Submit the above submit file. Here I am calling it ``MyJupyterLab.sh``

``sbatch MyJupyterLab.sh``

4. Get the URL from the SLURM output file.

Wait until the job gets resources allocated. Check the SLURM output file; when the job has resources allocated it will have a number of URLs inside at the bottom.

The SLURM output file is as default named ``slurm-<job-id>.out`` where you get the ``<job-id>`` when you submit the SLURM submit file (as in item 3. here).

**NOTE**: Grab the URL with the *hostname* since the localhost one requires you to login to the compute node and so will not work!

The file will look **similar** to this:

.. admonition:: 
   :class: dropdown

   .. code-block:: sh
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

 
To access the server, go to

``file:///.local/share/jupyter/runtime/jpserver-<newest>-open.html``

from a browser within the ThinLinc session. <newest> is a number that you find by looking in the directory ``.local/share/jupyter/runtime/`` under your home directory.

Or, to access the server you can copy and paste the URL from the file that is SIMILAR to this:

.. code-block:: sh

   http://b-cn1520.hpc2n.umu.se:8888/lab?token=c45b36c6f22322c4cb1e037e046ec33da94506004aa137c1

**NOTE** of course, do not copy the above, but the similar looking one from the file you get from running the batch script!!!

5. Start a webbrowser within HPC2N (ThinLinc interface). Open the html or put in the URL you grabbed, including the token:

.. figure:: ../img/jupyterlab-start.png
   :width: 450
   :align: center

After a few moments JupyterLab starts up:

.. figure:: ../img/jupyterlab_started.png
   :width: 450
   :align: center

You shut it down from the menu with "File" > "Shut Down"

LUNARC
######

See the `Desktop on Demand <https://uppmax.github.io/HPC-python/day1/ondemand-desktop.html>`_ section for this. 

NSC
### 

Through ThinLinc
^^^^^^^^^^^^^^^^

1. Login with ThinLinc (https://www.nsc.liu.se/support/graphics/) 

   - Download the client matching your local computer's OS and install it.
   - Start the ThinLinc client.
   - Change the “Server” setting to “tetralith.nsc.liu.se”.
   - Change the “Name” setting to your Tetralith username (e.g x_abcde).
   - Enter your cluster Tetralith password in the “Password” box.
   - Press the “Connect” button.
   - If you connect for the first time, you will see the “The server’s host key is not cached …” dialog. 

2. Load a JupyterLab module

   - Open a terminal    
   - This is an example for JupyterLab 4.2.0
   - ``module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0``   

3. Start JupyterLab

   - Type ``jupyter-lab`` in the terminal 
   - It will show some text, including telling you to open a url in a browser (inside ThinLinc/on Tetralith). If you just wait, it will open a browser with Jupyter.   

   - It will look similar to this: 

   .. figure:: ../img/jupyter-thinlinc-nsc.png

On your own computer through SSH tunneling 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Either do a regular SSH or use ThinLinc to connect to tetralith (change to your own username): 

   ``ssh x_abcde@tetralith.nsc.liu.se``

2. Change to your working directory

   ``cd <my-workdir>``

3. Load a module with JupyterLab in (here JupyterLab 4.2.0) 

   - module load buildtool-easybuild/4.8.0-hpce082752a2 GCC/13.2.0 Python/3.11.5 SciPy-bundle/2023.11 JupyterLab/4.2.0

4. Start jupyter with the no-browser flag

   - jupyter-lab --no-browser

   - You get something that looks like this: 

   .. figure:: ../img/jupyter-no-browser-nsc.png

Where I have marked a line with relevant info. Note that the port will change. 

5. Open a second terminal, on your home computer. Input this: 

   - ``ssh -N -L localhost:88XX:localhost:88XX x_abcde@tetralith1.nsc.liu.se``

   where you change 88XX to the actual port you got, and the name to your username. In my example it would be: 

   - ``ssh -N -L localhost:8867:localhost:8867 x_birbr@tetralith1.nsc.liu.se``

   .. figure:: ../img/local-ssh-to-nsc.png
   
6. Now grab the line that is similar to the one I marked in 4. and which has the same port as you used in 5. 

   - Input that line (url with token) in a browser on your local machine. You wil get something similar to this: 

   .. figure:: ../img/local-jupyter-lab.png


More information
################

- You can also check the lesson about `Jupyter on compute nodes <https://uppmax.github.io/R-python-julia-HPC/python/jupyter.html>`_ in our **Introduction to running R, Python and Julia in HPC workshop**)
- Documentation about `Jupyter on HPC2N <https://www.hpc2n.umu.se/resources/software/jupyter>`_
- Documentation about `Jupyter on UPPMAX <http://docs.uppmax.uu.se/software/jupyter/>`_


VScode
------

Spyder
------

Spyder is a powerful and flexible IDE originally developed to be the main scripting environment for scientific Anaconda users. It is designed to enable quick and easily repeatable experimentation, with automatic syntax checking, auto-complete suggestions, a runtime variable browser, and a graphics window that makes plots easy to manipulate after creation without additional code.

Spyder is available independent of Anaconda, but conda is still the recommended installer. Packages from the ``conda-forge`` source repo are still open-source, so conda is still usable on some facilities despite the recent changes in licensing. It is also possible to `build a pip environment with Spyder <https://docs.spyder-ide.org/current/installation.html#using-pip>`_, although this is only recommended for experienced Python users running on Linux operating systems.

To use Spyder on one of the HPC center resources, you must have a Thinlinc window open and logged into your choice of HPC resource. For personal use, it is relatively easy to `install as a standalone package on Windows or Mac <https://docs.spyder-ide.org/current/installation.html>`_, and there is also the option of `using Spyder online via Binder <https://mybinder.org/v2/gh/spyder-ide/binder-environments/spyder-stable?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fspyder-ide%252FSpyder-Workshop%26urlpath%3Ddesktop%252F%26branch%3Dmaster>`_.

.. tabs::

   .. tab:: LUNARC (Cosmos)

      On LUNARC, the recommended way to use Spyder is to use the On-Demand version in the Applications menu, under ``Applications - Python``. All compatible packages should be configured to load upon launching, so you should only have to specify walltime and maybe a few extra resource settings with the GfxLauncher so that spyder will run on the compute nodes. Refer to `the Desktop On Demand documentation <https://uppmax.github.io/HPC-python/day1/ondemand-desktop.html>`_ to help you fill in GfxLauncher prompt.

      Avoid launching Spyder from the command line on the login node.

   .. tab:: HPC2N (Kebnekaise)

      The only available version of Spyder on Kebnekaise is Spyder/4.1.5 for Python-3.8.2 (the latest release of Spyder available for users to install in their own environments is 6.0.2). Python 3.8.2 is associated with compatible versions of Matplotlib and Pandas, but not Seaborn or any of the ML packages to be covered later. To run the available version of Spyder, run the following commands:

      .. code-block:: console

         ml GCC/9.3.0  OpenMPI/4.0.3  Python  Spyder
         spyder3
      
      If you want a newer version with more and newer compatible Python packages, you will have to create a virtual environment.

   .. tab:: UPPMAX (Rackham)

      Spyder is not available centrally on Rackham. It will have to be installed in your (conda) virtual environment.

   .. tab:: NSC (Tetralith)

      Spyder is not available on Tetralith. It will have to be installed in your (conda) virtual environment.


Features
^^^^^^^^

When you open Spyder, you should see something like the figure below. There should be a large pane on the left for code, and two smaller panes on the right. Each of the 3 panes have their own button with 3 horizontal lines (the menu button or "burger icon") in the top right, each with additional configuration options for those panes.

.. image:: ./docs/img/cosmos-on-demand-spyder.png
   :width: 800 px

The top right pane has several useful tabs.

* **Help** displays information about commands and starts up with a message instructing you on how to use it.
* **Variable explorer** shows a table of currently defined variables, their datatypes, and their current values at runtime. It updates every time you either run a script file or run a command in the IPython console.
* **Files** shows the file tree for your current working directory
* Depending on the version , there may also be a **Plots** tab for displaying and adjusting graphics produced with, e.g. Matplotlib. 

You can move any of these tabs into separate windows by clicking the menu (burger) button and selecting "Undock". This is especially helpful for the plotting tab.

The bottom right pane shows the IPython console and a history tab. The IPython console is your Python command line, and runs in your current working directory unless changed with ``os.chdir()``. The default path is whatever directory you were in when you launched Spyder, but that can be changed in Preferences. If you run a script file that you've saved to a different directory using the green arrow icon on the menu ribbon, the IPython console will switch your working directory to the one containing that script. The history tab stores the last 500 lines of code excuted in the IPython console.

Most of the icons along the top menu bar under the verbal menu are running and debugging commands. You can hover over any of them to see what they do.


Configuring Spyder
^^^^^^^^^^^^^^^^^^

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

.. image:: ./docs/img/cosmos-on-demand-spyder-preferences.png
   :width: 800 px

Now, graphics should appear in their own popup that has menu options to edit and save the content.

