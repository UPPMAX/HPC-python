Prepare the environment
=======================

.. admonition:: Goal

    The goal of this page to make sure you can follow the course.

These are the things you need to follow the course:

- you can log in to at least one HPC cluster, in at least one way
- you can start a text editor

These are discussed in detail below

.. note::

   - There will be an opportunity to get help with log in every morning of the workshop at 9:00.
   - You are also welcome to join the On-boarding at 13:00 the day before the ordinary program starts.

Log in to one of the HPC systems covered in this course
-------------------------------------------------------

.. admonition:: To be done before

   - Follow the steps in the emailed instructions.
   - First time you need to use a terminal to set password
   - When password is set you can begin to use ThinLinc as well.


.. warning::

   - When logging in to UPPMAX the first time in ThinLinc, choose XFCE desktop. 
   - On HPC2N, you will use the MATE desktop as default. 
   - Whe logging in to LUNARC the first time in ThinLinc, choose GNOME Classis Desktop.  
   - On NSC you will use XFCE desktop as default. 

.. warning::

   - When you login to Cosmos, whether through ThinLinc or regular SSH client, you need 2FA 
     
      - https://lunarc-documentation.readthedocs.io/en/latest/getting_started/login_howto/
      - https://lunarc-documentation.readthedocs.io/en/latest/getting_started/authenticator_howto/

.. warning::

   - When you login to Tetralith, whether through ThinLinc or regular SSH client, you need 2FA 

      - https://www.nsc.liu.se/support/2fa/ 

.. seealso::

   - `Log in to Rackham <http://docs.uppmax.uu.se/getting_started/login_rackham/>`_ 
   - `Log in to Kebnekaise <http://docs.uppmax.uu.se/getting_started/login_rackham/>`_ 

   - `Log in to Cosmos <https://lunarc-documentation.readthedocs.io/en/latest/getting_started/login_howto/>`_

   - `Log in to Tetralith <https://www.nsc.liu.se/support/getting-started/>`_ 

       - `Log in to Tetralith with graphical applications (scroll down for ThinLinc) <https://www.nsc.liu.se/support/graphics/>`_ 

These are the ways to access your HPC cluster and how that looks like:

+---------------------------------------------+-------------------------------------------------------------------+
| How to access your HPC cluster              | How it looks like                                                 |
+=============================================+===================================================================+
| Remote desktop via a website                | .. figure:: img/rackham_remote_desktop_via_website_480_x_270.png  |
+---------------------------------------------+-------------------------------------------------------------------+
| Remote desktop via a local ThinLinc client  | .. figure:: img/thinlinc_local_rackham_zoom.png                   |
+---------------------------------------------+-------------------------------------------------------------------+
| Console environment using an SSH client     | .. figure:: img/login_rackham_via_terminal_terminal_409_x_290.png |
+---------------------------------------------+-------------------------------------------------------------------+

These are the ways to access your HPC cluster and some of their features:

+---------------------------------------------+-------------------------------------------------------------------+
| How to access your HPC cluster              | Features                                                          |
+=============================================+===================================================================+
| Remote desktop via a website                | Familiar remote desktop                                           |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Clumsy and clunky                                                 |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Slow on UPPMAX                                                    |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | No need to install software                                       |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Works on HPC2N and UPPMAX                                         |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Needs 2FA for UPPMAX                                              |
+---------------------------------------------+-------------------------------------------------------------------+
| Remote desktop via a local ThinLinc client  | Familiar remote desktop                                           |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Clumsy                                                            |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Need to install ThinLinc                                          |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Works on all centers                                              |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Needs 2FA for UPPMAX                                              |
+---------------------------------------------+-------------------------------------------------------------------+
| Console environment using an SSH client     | A console environment may be unfamiliar                           |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Great to use                                                      |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Need to install an SSH client                                     |
+---------------------------------------------+-------------------------------------------------------------------+
|                                             | Works on all centers                                              |
+---------------------------------------------+-------------------------------------------------------------------+

Here is an overview of where to find the documentation and a video showing the procedure:

+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| HPC Center | Method                 | Documentation                                                                                          | Video                                                      |
+============+========================+========================================================================================================+============================================================+
| HPC2N      | SSH                    | `here <https://docs.hpc2n.umu.se/documentation/access/>`_                                              | `here <https://youtu.be/pIiKOKBHIeY?si=2MVHoFeAI_wQmrtN>`_ |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| HPC2N      | Local ThinLinc client  | `here <https://docs.hpc2n.umu.se/documentation/access/>`_                                              | `here <https://youtu.be/_jpj0GW9ASc?si=1k0ZnXABbhUm0px6>`_ |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| HPC2N      | Remote desktop website | `here <https://docs.hpc2n.umu.se/documentation/access/>`_                                              | `here <https://youtu.be/_O4dQn8zPaw?si=z32av8XY81WmfMAW>`_ |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| LUNARC     | SSH                    | `here <https://lunarc-documentation.readthedocs.io/en/latest/getting_started/login_howto/>`_           | `here <https://youtu.be/sMsenzWERTg>`_                     |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| LUNARC     | Local ThinLinc client  | `here <https://lunarc-documentation.readthedocs.io/en/latest/getting_started/using_hpc_desktop/>`_     | `here <https://youtu.be/wn7TgElj_Ng>`_                     |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| UPPMAX     | SSH                    | `here <https://docs.uppmax.uu.se/getting_started/login_rackham_remote_desktop_local_thinlinc_client>`_ | `here <https://youtu.be/TSVGSKyt2bQ>`_                     |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| UPPMAX     | Local ThinLinc client  | `here <https://docs.uppmax.uu.se/getting_started/login_rackham_console_password/>`_                    | `here <https://youtu.be/PqEpsn74l0g>`_                     |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+
| UPPMAX     | Remote desktop website | `here <https://docs.uppmax.uu.se/getting_started/login_rackham_remote_desktop_website/>`_              | `here <https://youtu.be/HQ2iuKRPabc>`_                     |
+------------+------------------------+--------------------------------------------------------------------------------------------------------+------------------------------------------------------------+

Need help? Contact support:

+------------+-----------------------------------------------------------------------+
| HPC Center | How to contact support                                                |
+============+=======================================================================+
| HPC2N      | `Contact HPC2N support <https://docs.hpc2n.umu.se/support/contact/>`_ |
+------------+------------------------+----------------------------------------------+
| LUNARC     | `Contact LUNARC support <https://www.lunarc.lu.se/getting-help/>`_    |
+------------+------------------------+----------------------------------------------+
| UPPMAX     | `Contact UPPMAX support <https://docs.uppmax.uu.se/support/>`_        |
+------------+------------------------+----------------------------------------------+

.. keypoints::

   - When you log in from your local computer you will always arrive at a login node with limited resources. 
       - You reach the calculations nodes from within the login node (See  Submitting jobs section)
   - You reach UPPMAX/HPC2N/LUNARC clusters either using a terminal client or Thinlinc
   - Graphics are included in Thinlinc and from terminal if you have enabled X11.
   - Which client to use?
       - Graphics and easy to use
       - ThinLinc
   - Best integrated systems
       - Visual Studio Code has several extensions (remote, SCP, programming IDE:s)
       - Windows: MobaXterm is somewhat easier to use.

Text editors on the Clusters
----------------------------
- Nano
- gedit
- mobaxterm built-in

.. seealso::

   - http://docs.uppmax.uu.se/software/text_editors/
   - https://docs.hpc2n.umu.se/tutorials/linuxguide/#editors 

.. hint::

   - There are many ways to edit your scripts.
   - If you are rather new.

      - Graphical: ``$ gedit <script> &`` 
   
         - (``&`` is for letting you use the terminal while editor window is open)

         - Requires ThinLinc or ``ssh -Y ...`` or ``ssh -X``

      - Terminal: ``$ nano <script>``

   - Otherwise you would know what to do!
   - |:warning:| The teachers may use their common editor, like ``vi``/``vim``
      - If you get stuck, press: ``<esc>`` and then ``:q`` !
 

.. demo::

   - Let's make a script with the name ``example.py``  

   .. code-block:: console

      $ nano example.py

   - Insert the following text

   .. code-block:: python

      # This program prints Hello, world!
      print('Hello, world!')

   - Save and exit. In nano: ``<ctrl>+O``, ``<ctrl>+X``

   You can run a python script in the shell like this:

   .. code-block:: console

      $ python example.py
      # or 
      $ python3 example.py

