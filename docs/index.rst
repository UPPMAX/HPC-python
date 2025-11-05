.. Python-at-UPPMAX documentation master file, created by
   sphinx-quickstart on Fri Jan 21 18:24:04 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to "Using Python in an HPC environment" course material
===============================================================

.. admonition:: Content

   - This course aims to give a brief, but comprehensive introduction to using Python in an HPC environment.
   - You will learn how to
      - use modules to load Python
      - find site installed Python packages
      - install packages yourself
      - use isolated environments,
      - write a batch script for running Python
      - use Python in parallel
      - use Python for ML and on GPUs.
   - This course will consist of lectures interspersed with hands-on sessions where you get to try out what you have just learned.
   - **We aim to give this course in spring and fall every year.**

.. admonition:: **Course learning outcomes**

    - You can find and load a Python package modules
    - You can determine if a Python package is installed
    - You can use an isolated environment
    - You can install Python packages
    - You can write and submit batch job scripts
    - You can use the compute nodes interactively
    - You can use a Python IDE
    - Use Python in parallel
    - Use Python for ML
    - Use GPUs with Python

Schedule Autumn 2025
--------------------

.. admonition:: Schedule

    See :ref:`schedule`

.. warning::

   **Not covered**

    - Improve Python *coding* skills
    - Specifics of other clusters

.. admonition:: Course approach to deal with multiple HPC centers

   **Target group**

   - The course is for present or presumptive users at the **NAISS** resources at PDC and NSC as well as UPPMAX, HPC2N, LUNARC, or possibly other clusters in Sweden.
   - The course is a NAISS cooperation with **teachers** from **UPPMAX** (Rackham, Snowy, Bianca), **HPC2N** (Kebnekaise), and **LUNARC** (Cosmos) and will focus on systems at the Swedish academic HPC centres with NAISS personnel.
   - Although there are differences we will only have **few seperate sessions**.
   - Most participants will use The NAISS resources **Tetralith** system at NSC or **Dardel** system at PDC for the course.

       - Alvis at **C3SE (Chalmers)** may be added in the future.

   - Users with the "local" affiliation below, can work at the following clusters:

      - Kebnekaise: UmU, IRF, MIUN, SLU, LTU.
      - Cosmos: LU.
      - Rackham/Snowy: UU.

   - The general information given in the course will be true for all/most HPC centres in Sweden.

      - The examples will often have specific information, like module names and versions, which may vary. What you learn here should help you to make any changes needed for the other centres.
      - When present, links to the Python documentation at other NAISS centres are given in the corresponding session.

.. admonition:: Cluster documentation of Python

   - Documentation at the HPC centres
      - `UPPMAX <https://docs.uppmax.uu.se/software/python/>`_
      - `HPC2N <https://docs.hpc2n.umu.se/tutorials/userinstalls/#python__packages>`_
      - `LUNARC <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`_
      - `NSC <https://www.nsc.liu.se/support/>`_
      - `PDC <https://support.pdc.kth.se/doc/applications/python/>`_

.. admonition:: Some practicals

   - `Code of Conduct <https://github.com/UPPMAX/R-matlab-julia-HPC/blob/main/CODE_OF_CONDUCT.md>`_
      - Be nice to each other!

   - Zoom
       - You should have gotten an email with the links

       - Zoom policy:

           - Zoom chat (maintained by co-teachers):

                - technical issues of zoom
                - technical issues of your settings
                - direct communication
                - each teacher may have somewhat different approach

            - collaboration document (see below):

                - "explain again"
                - elaborating the course content
                - solutions for your own work

       - Recording policy:
           - **All lectures and demos** will be available on HPC2Ns **Youtube** channel after the course.

           - Due to different teaching styles, the teacher of a session decides if a lesson will be:
               - recorded without learners
               - recorded live with learners

           - The teachers will be clear if it is recorded or not and it will be visible in the ZOOM.
           - For the live-recordings

               - The questions asked per microphone during these sessions will be recorded
               - If you don't want your voice to appear:

                   - use the collaboration document (see below)

       - The Zoom main room is used for most lectures
       - Some sessions use breakout rooms for exercises, some of which use a silent room

.. admonition:: Q/A collabration document

   - Use the Q/A page for the workshop with your questions.
       - `https://umeauniversity.sharepoint.com/:w:/s/HPC2N630/EZdjaHPgG-NMl9luOFWbQacBsYpY8zs2YmgosHrMcOQnuQ?e=yEGirA <https://umeauniversity.sharepoint.com/:w:/s/HPC2N630/EZdjaHPgG-NMl9luOFWbQacBsYpY8zs2YmgosHrMcOQnuQ?e=yEGirA>`_
   - Use this page for the workshop with your questions
   - It helps us identify content that is missing in the course material
   - We answer those questions as soon as possible

.. warning::

   - Please be sure that you have gone through the :ref:`prereqs`
      - It mentions the familiarity with the LINUX command line.
      - The applications to connect to the clusters: terminals and ThinLinc (remote graphical desktop)
   - This course does not aim to improve your coding skills. Rather you will learn to understand the ecosystems and navigations for the the different languages on a HPC cluster.


Content of the course
---------------------

.. toctree::
   :maxdepth: 2
   :caption: Pre-requirements:

   prereqs.rst
   preparations.rst

.. toctree::
   :maxdepth: 2
   :caption: Common:

   schedule.rst
   common/login.rst
   common/use_tarball.rst
   common/use_text_editor.rst
   common/understanding_clusters.rst
   common/naiss_projects_overview.rst
   common/interactive_ondemand.md

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 1 (Intro to Python):

   common/day1.rst

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 2 (packages and analysis):

   day2/intro.rst
   day2/use_packages.rst
   day2/install_packages.rst
   day2/use_isolated_environments.rst
   day2/interactive.md
   day2/ondemand-desktop.rst
   day2/IDEs.rst
   summary2.rst
   day2/evaluation.rst
   day2/python_at_hpc_centers.rst

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 3 (advanced analysis):

   day3/batch.md
   day3/big_data.md
   summary3.rst
   day3/evaluation.rst

.. toctree::
   :maxdepth: 2
   :caption: Lessons day 4 (parallel and ML):

   day4/parallel.rst
   day4/gpu.md
   day4/ml.md
   day4/dim_reduction.rst
   day4/evaluation.rst
   summary4.rst

.. toctree::
   :maxdepth: 2
   :caption: Extra:

   bianca.rst
   evaluations/README.md
   extra/other_courses.rst
   extra/packages_deeper.rst
   extra/isolated_deeper.rst
   extra/jupyterHPC2N.rst
   kebnekaise.md
   day4/ML_deeper.rst
   reflections/README.md
   uppmax.rst

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
