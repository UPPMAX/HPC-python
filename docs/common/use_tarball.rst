.. _common-use-tarball:

Use the tarball with exercises
==============================

.. admonition:: Goal

    - You can run the example files needed for the exercises
    - If any, you are able to navigate to the project folder
    - If needed, you are able to create a subfolder in the project folder

A tarball is a file that contains multiple files,
similar to a zip file.
To use the files it contains, it needs to be untarred/unzipped/uncompressed
first.

Procedure
---------

.. admonition:: Prefer a video?
    :class: dropdown

    +------------+------------------------------------------------------------+
    | HPC cluster| Link to YouTube video                                      |
    +============+============================================================+
    | Alvis      | `Here <https://youtu.be/o1K8YuYUfGA>`__                    |
    +------------+------------------------------------------------------------+
    | COSMOS     | `Here <https://youtu.be/lYyzNzX0pww>`__                    |
    +------------+------------------------------------------------------------+
    | Dardel     | ``TODO``                                                   |
    +------------+------------------------------------------------------------+
    | Kebnekaise | ``TODO``                                                   |
    +------------+------------------------------------------------------------+
    | Rackham    | ``TODO``                                                   |
    +------------+------------------------------------------------------------+
    | Tetralith  | ``TODO``                                                   |
    +------------+------------------------------------------------------------+

Below are the steps to download and extract the files needed
for exercises in the course.

Step 1: if there is a project folder, navigate to your project folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step only applies if your HPC cluster uses a project folder.
Skip this step if your HPC cluster does not use project folders.

.. admonition:: Forgot if your HPC cluster uses a project folder?
    :class: dropdown

    In that case, do the rest of this step. You'll find out
    how to navigate to the right folder in the answers.

On your favorite HPC cluster, navigate to the project folder
of this course. See :ref:`common-naiss-projects-overview` for where these are.

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    +------------+------------------------------------------------------------+
    | HPC cluster| Course project                                             |
    +============+============================================================+
    | Alvis      | None. Use ``cd /mimer/NOBACKUP/groups/[your_project_code]``|
    +------------+------------------------------------------------------------+
    | Bianca     | Use ``cd /proj/[your_project_code]``                       |
    +------------+------------------------------------------------------------+
    | COSMOS     | Use home folder (instead of a project folder): ``cd ~``    |
    +------------+------------------------------------------------------------+
    | Dardel     | ``cd /cfs/klemming/projects/snic/hpc-python-spring-naiss`` |
    +------------+------------------------------------------------------------+
    | Kebnekaise | ``cd /proj/nobackup/hpc-python-spring``                    |
    +------------+------------------------------------------------------------+
    | LUMI       | None. Use ``cd /project/[your_project_code]``              |
    +------------+------------------------------------------------------------+
    | Rackham    | ``cd /proj/hpc-python-uppmax``                             |
    +------------+------------------------------------------------------------+
    | Tetralith  | ``cd /proj/hpc-python-spring-naiss/users/``                |
    +------------+------------------------------------------------------------+

.. admonition:: In general, how can I see which projects I have?
    :class: dropdown

    You can see your projects in `SUPR <https://supr.naiss.se/>`__.

    For some screenshots, see
    `the UPPMAX documentation <https://docs.uppmax.uu.se/getting_started/project/#view-your-uppmax-projects>`__.


Step 2: if there is a project folder, create a folder for yourself
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step only applies if your HPC cluster uses a project folder.
Skip this step if your HPC cluster does not use project folders.

In the project folder of this course, create a folder for yourself.
For example, if your name is Sven, create a folder called ``sven``

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    ``mkdir sven``

Step 3: if there is a project folder, navigate inside that folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step only applies if your HPC cluster uses a project folder.
Skip this step if your HPC cluster does not use project folders.

From the project folder of this course,
navigate inside your personal folder

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    ``cd sven``

Step 4: get the tarball
^^^^^^^^^^^^^^^^^^^^^^^

In the terminal, in your personal folder, type the following command:

``wget https://github.com/UPPMAX/HPC-python/raw/refs/heads/main/exercises.tar.gz``

Step 5: Uncompress the tarball
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the terminal, in your personal folder, type the following command:

``tar -xvzf exercises.tar.gz``

Step 6: Navigate in the folder of that day
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After decompressing, there is a folder called  ``day2``, or ``day3`` or ``day4``
that contains the exercises. Navigate into that folder.

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command for day 2:

    ``cd day2``
