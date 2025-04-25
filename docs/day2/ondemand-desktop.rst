Desktop On Demand
=================


.. questions::

   - What is Desktop On Demand?
   - When should I use it?
   - Where and how do I start?

   
.. objectives:: 

   - Short introduction to Desktop On Demand
   - Typical setup and usage

    

What is Desktop On Demand?
--------------------------
At LUNARC (and coming to other HPC centers in the not-too-distant future), the Desktop On Demand service provides an interactive environment for submitting programs to SLURM without the typical shell script. It uses the graphical application launcher (GfxLauncher) to submit jobs to SLURM, connect to the application running on the Compute node, and monitor the application's progress. It is not itself an option in the Applications menu; rather, it's the engine that lets you run the listed applications interactively.

Later you will hear us talking more about the GfxLauncher than Desktop On Demand because while the latter is the underlying service provider, the former is what you will see every time you start an interactive application.

Both Spyder and Jupyter are available through Desktop On Demand. Other applications can be launched from an Interactive Terminal or Accelerated Terminal session. VSCode currently only runs on the login nodes, but its command line can be configured to connect and submit jobs to the compute nodes.

Desktop On Demand requires the use of the Thinlinc interface. It is not accessible via ssh. 

.. warning:: 

   Dardel also has On-Demand applications located in the equivalent place on its remote desktop, but only 30 ThinLinc licenses are available for the whole facility. Talks are ongoing about whether and by how much to increase the number of licenses, but until the changes are implemented, we advise using SSH with -X forwarding instead. In our experience, ThinLinc access to Dardel is not reliably available, and even when connection succeeds, queue times for On-Demand applications can be very long.


When should I use it?
---------------------
Desktop On Demand is most appropriate for *interactive* work requiring small-to-medium amounts of computing resources.

The GfxLauncher will prompt you for resource specification and then Desktop On Demand will put your resource requests into the same SLURM queue as every other job. Depending on the time and resources you request, you may have to wait a while for your session to start. For information about the capabilities of the available nodes, you can explore the `LUNARC homepage's Systems tab <https://www.lunarc.lu.se/systems/>`_ and follow the links to your desired resource. 

.. admonition:: **Wall Time Limits**
   
      Wall time for interactive work with Desktop On Demand is restricted to 48 consecutive
      (not business) hours. In practice, there can be significant startup delays for wall times
      as short as 4 hours. Users should save their work frequently and be conservative in their
      wall time estimates. To save GPU resources, we also encourage users who are able to submit 
      jobs requiring minimal supervision as ordinary batch scripts to do that whenever feasible.


Some On Demand applications will let you configure and submit separate batch jobs that are not bound by the parameters set for the graphical user interface (GUI) in GfxLauncher, although the initial configuration process can be rather involved.

Getting Started
---------------

Where are the On-Demand Applications?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the LUNARC HPC Desktop, the Applications menu lists all the applications available to be run interactively, and opening one prefixed by "Applications -" will start it in Desktop On Demand. There is no specific "Desktop On Demand" application in the list. Most common IDEs can be found in a drop-down menu that appears when you hover over ``Applications - <Language>`` for your language of choice, but more niche packages may be listed under a subject-matter field instead, like ``Applications - Engineering``.

.. admonition:: Terminals on the Compute nodes

   If you don't see the program you want to run interactively listed under any other ``Applications`` sub-menus, you may still be able to launch it via one of the terminals under ``Applications - General``, or the **GPU Accelerated Terminal** (GPU support) under ``Applications - Visualization``.  
   
   
   .. figure:: ../img/Cosmos-AppMenu.png
      :width: 400
      :align: center

   
   The CPU terminal allows for a wall time of up to 168 hours (7 days), while the two GPU terminals can only run for 48 hours (2 days) at most. For more on the specifications of the different nodes these terminals can run on, see `LUNARC's webpage on COSMOS <https://www.lunarc.lu.se/systems/cosmos/>`_.

**Please be aware that only the applications in the menus prefixed with "Applications -" are set up to run on the Compute nodes.** If you start a terminal session or other application from ``Favorites`` or ``System Tools`` and launch an interactive program from that, it will run on a Login node, with all the risks that that entails for your user privileges.


.. note:: What if On-Demand Applications are missing from the menu?
   :class: dropdown

   On rare occasions, a user may find that the Applications menu is missing all ``Applications - <App_group>`` options. This usually indicates that your ``.bashrc`` file is either missing or has had problematic changes made to it, especially if LMOD commands like ``ml spider <package>`` are also not recognized. If you are a new user on your very first session on COSMOS, the problem should resolve itself if you start a new ThinLinc session with "End existing session" selected. If you are not a new user and module commands are recognized, running ``gfxmenu --force`` in a terminal session may resolve the issue; otherwise, you will probably have to submit a support ticket.


How do I start?
^^^^^^^^^^^^^^^

For most programs, the start-up process is roughly the same:

#. Log into COSMOS via Thinlinc to start a LUNARC HPC Desktop session.
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


