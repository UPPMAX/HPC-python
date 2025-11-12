Desktop On Demand
=================

.. objectives::

   You will learn:

   - What is On-Demand and when to use it
   - Which interface to use on each resource and how to start them
   - How to set the job parameters for your application
    

What is Desktop On Demand? Is it right for my job?
--------------------------------------------------

On Cosmos (LUNARC), Kebnekaise (HPC2N), and Dardel (PDC), some applications are available through one of a couple of On Demand services. On Demand applications provide an interactive environment to schedule jobs on compute nodes using a graphic user interface (GUI) instead of the typical batch submission script. How you reach this interface is dependent on the system you use and their choice of On Demand client.

- **Cosmos** and **Dardel** use the On-Demand Desktop developed at LUNARC, which is accessible via Thinlinc.
- **Kebnekaise** uses Open OnDemand[#f1]_ via a `dedicated web portal, https://portal.hpc2n.umu.se <https://portal.hpc2n.umu.se>`__.

Desktop On-Demand is most appropriate for *interactive* work requiring small-to-medium amounts of computing resources. Non-interactive jobs and jobs that take more than a day or so should generally be submitted as batch jobs. If you have a longer job that requires an interactive interface to submit, make sure you keep track of the wall time limits for your facility.

On-Demand applications are *not* accessible via SSH; you must use either Thinlinc (Cosmos and Dardel) or the dedicated web portal (Kebnekaise).

.. important:: "On-Demand App Availability for this Course"

   - Jupyter (Lab and/or Notebook) is available as an On-Demand application at all 3 facilities covered on this page. For Cosmos specifically, it can also load custom conda environments (but NOT pip environments).
   - On Cosmos and Kebnekaise, VSCode can also be run via On-Demand.
   - Spyder can be run via On-Demand on Cosmos only. It also supports custom conda environments.
   - On Cosmos, there are also interactive On-Demand command lines (for CPUs and GPUs) under `Applications - General` that can be used to start Jupyter or Spyder with a custom pip-based environment.

.. warning:: 

   Dardel also has On-Demand applications located in the equivalent place on its remote desktop, but only 30 ThinLinc licenses are available for the whole facility. Talks are ongoing about whether and by how much to increase the number of licenses, but until the changes are implemented, we advise using SSH with -X forwarding instead. In our experience, ThinLinc access to Dardel is not reliably available, and even when connection succeeds, queue times for On-Demand applications can be very long.

Starting the On-Demand Interface
--------------------------------

.. tabs::

   ..tab:: "COSMOS (and Dardel)"
   
       For most programs, the start-up process is roughly the same:
   
       1. Log into COSMOS (or Dardel) via your usual Thinlinc client or browser interface to start an HPC Desktop session.
       2. Click ``Applications`` in the top left corner, hover over the items prefixed with ``Applications -`` until you find your desired application (on Dardel, On-Demand applications are prefixed with ``PDC-``), and click it. The top-level Applications menu on Cosmos looks like this:
          
       .. figure:: ../img/Cosmos-AppMenu.png
          :width: 400
          :align: center

       .. warning::
       
          If you start a terminal session or another application from ``Favorites``, ``System Tools``, or other menu headings not prefixed with ``Applications -`` or ``PDC-``, and launch an interactive program from that, it will run on a login node. Do not run intensive programs this way!

=== "Kebnekaise"

    To start an Open OnDemand session on Kebnekaise,
    
    1. Open `https://portal.hpc2n.umu.se <https://portal.hpc2n.umu.se>`__ in your browser. The page looks like this:
    
        <img src="https://docs.hpc2n.umu.se/images/open-ondemand-portal.png" alt="HPC2N Open On-Demand Portal" width="1200"/>
    
    2. Click the blue button labeled "Login to HPC2N OnDemand".
    3. A login window should open with boxes for your login credentials. Enter your HPC2N username and password, then click "Sign In".
    4. You will now be on the HPC2N Open OnDemand dashboard. The top of it looks like this:
    
        <img src="https://docs.hpc2n.umu.se/images/open-ondemand-dashboard.png" alt="HPC2N Open On-Demand Portal" width="1200"/>
    
    5. Find the ``Interactive Apps`` tab in the menu bar along the top and click it to open a drop-down menu of available apps. The menu currently looks like this:
    
        <img src="https://docs.hpc2n.umu.se/images/open-ondemand-apps.png" alt="HPC2N Open On-Demand dashboard" width="1200"/>
    
    !!! warning
    
        Unlike on Cosmos and Dardel, On-Demand applications on Kebnekaise are **not** reachable through Thinlinc, regardless of whether you use the desktop client or a browser! If you find similar-looking applications in the Thinlinc interface, be aware that they all run on login nodes!


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


.. [#f1] Open OnDemand is a web service that allows HPC users to schedule jobs, run notebooks and work interactively on a remote cluster from any device that supports a modern browser. The Open OnDemand project was funded by NSF and is currently maintained by the Ohio SuperComputing Centre. Read more about `OpenOndemand.org <https://openondemand.org/>`__.
