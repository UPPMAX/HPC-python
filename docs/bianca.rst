On Bianca cluster
-----------------

.. questions::

   - What is Bianca?
   - How to install packages on Bianca?
   
.. objectives:: 

   - Overview of pros with CONDA on Bianca
   - Show how to pip install if package is not available from conda

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

   - First try: Conda
   - Second try: pip install or isolated environment moved from Rackham to Bianca via secure transfer
   
