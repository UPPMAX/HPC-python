.. _devel_iso:

Developing in isolated environments
===================================

- You may want to develop python programs and/or packages yourself. This is not the focus for this course.
- However, below we present how one can use the isolated environment to spread your environment to colleagues or others who will use your software/scrips.

You may have a look on these pages

`Dependencies <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_

.. keypoints::
   - Install dependencies by first recording them in ``requirements.txt`` or ``environment.yml`` and install using these files, then you have a trace.
   - Use isolated environments and avoid installing packages system-wide.

`Packaging <https://aaltoscicomp.github.io/python-for-scicomp/packaging/>`_

.. keypoints::

   - It is worth it to organize your code for publishing, even if only you are using it.

   - PyPI is a place for Python packages

   - conda is similar but is not limited to Python

Creator/developer
.................

- First _create_ and _activate_ an environment (see above)
- Install packages with pip
- Create file from present virtual environment:

.. code-block:: console

   $ pip freeze > requirements.txt

- That includes also the *system site packages* if you included them with ``--system-site-packages``
- Test that everything works by running use cases scripts within the environment
- You can list packages specific for the virtualenv by ``pip list --local`` 

- So, creating a file from just the local environment:

.. code-block:: console

   $ pip freeze --local > requirements.txt

.. note:: 

   ``requirements.txt`` (used by the virtual environment) is a simple text file which looks similar to this::

      numpy
      matplotlib
      pandas
      scipy

   ``requirements.txt`` with versions that could look like this::

      numpy==1.20.2
      matplotlib==3.2.2
      pandas==1.1.2
      scipy==1.6.2

- Deactivate

User
....

- Create an environment based on dependencies given in an environment file
- This can be done in new virtual environment or as a genera installation locally (not activating any environment
  
.. code-block:: console

   pip install -r requirements.txt

- Check

.. code-block:: console

   pip list
   
.. admonition:: More on dependencies

   - `Dependency management from course Python for Scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_


