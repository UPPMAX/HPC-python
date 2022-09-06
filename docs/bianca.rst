On Bianca cluster
-----------------

.. questions::

   - What syntax is used to make a lesson?
   - How do you structure a lesson effectively for teaching?

   ``questions`` are at the top of a lesson and provide a starting
   point for what you might learn.  It is usually a bulleted list.
   (The history is a holdover from carpentries-style lessons, and is
   not required.)
   
.. objectives:: 

   - Show how to load Python
   - show how to run Python scripts and start the Python commandline

.. Note::

    Bianca has no Internet! ``pip`` will not work!
    
    Since we have mirrored conda repositories locally `conda` will work also on Bianca!


- First try conda, as above.


- If packages are not available, follow the guideline below, while looking at https://uppmax.uu.se/support-sv/user-guides/bianca-user-guide and https://www.uppmax.uu.se/support/user-guides/transit-user-guide/.


- Make an installation on Rackham and then use the wharf to copy it over to your directory on Bianca.

  - Path on Rackham and Bianca could be (~/.local/lib/python<version>/site-packages/ ). 

- You may have to:

  - in source directory:

     .. prompt:: bash $

        cp –a <package_dir> <wharf_mnt_path>
	
- You may want to ``tar`` before copying to include all possible symbolic links:

   .. prompt:: bash $

      tar cfz <tarfile.tar.gz> <package> 	
	
- and in target directory (``wharf_mnt``) on Bianca:
    
   .. prompt:: bash $

      tar xfz <tarfile.tar.gz> #if there is a tar file!		
      mv –a  <file(s)> ~/.local/lib/python<version>/site-packages/ 

.. keypoints::

   - What the learner should take away
   - point 2
   
