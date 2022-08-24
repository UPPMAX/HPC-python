Exercises and demos
============================

Examples
--------

Isolated
########



.. admonition:: Load modules for Python, numpy (in SciPy-bundle), activate the environment, and install spacy on Kebnekaise at HPC2N 
    :class: dropdown
   
        .. code-block:: sh
           
           b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
           b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ source vpyenv/bin/activate
           (vpyenv) b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ pip install --no-cache-dir --no-build-isolation spacy 
   
2) Installing seaborn. Using existing modules for numpy (in SciPy-bundle), matplotlib, and the vpyenv we created under Python 3.9.5. Note that you need to load Python again if you have been logged out, etc. but the virtual environment remains, of course   

.. admonition:: Load modules for Python, numpy (in SciPy-bundle), matplotlib, activate the environment, and install seaborn on Kebnekaise at HPC2N 
    :class: dropdown
   
        .. code-block:: sh
           
           b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05 matplotlib/3.4.2
           b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ source vpyenv/bin/activate
           (vpyenv) b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ pip install --no-cache-dir --no-build-isolation seaborn 

Using the vpyenv created earlier and the spacy we installed under example 1) above. 

.. admonition:: Load modules for Python, numpy (in SciPy-bundle), activate the environment (on Kebnekaise at HPC2N) 
    :class: dropdown
   
        .. code-block:: sh
           
           b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
           b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ source vpyenv/bin/activate
           (vpyenv) b-an01 [/proj/nobackup/support-hpc2n/bbrydsoe]$ python
           Python 3.9.5 (default, Jun  3 2021, 02:53:39) 
           [GCC 10.3.0] on linux
           Type "help", "copyright", "credits" or "license" for more information.
           >>> import spacy
           >>> 

Interactive
###########

.. admonition:: Example, Requesting 4 cores for 30 minutes, then running Python 
    :class: dropdown
   
        .. code-block:: sh

            b-an01 [~]$ salloc -n 4 --time=00:30:00 -A SNIC2022-22-641
            salloc: Pending job allocation 20174806
            salloc: job 20174806 queued and waiting for resources
            salloc: job 20174806 has been allocated resources
            salloc: Granted job allocation 20174806
            salloc: Waiting for resource configuration
            salloc: Nodes b-cn0241 are ready for job
            b-an01 [~]$ module load GCC/10.3.0 OpenMPI/4.1.1 Python/3.9.5
            b-an01 [~]$ 

.. admonition:: Adding two numbers from user input (add2.py)
    :class: dropdown
   
        .. code-block:: python

            # This program will add two numbers that are provided by the user
            
            # Get the numbers
            a = int(input("Enter the first number: ")) 
            b = int(input("Enter the second number: "))
            
            # Add the two numbers together
            sum = a + b
            
            # Output the sum
            print("The sum of {0} and {1} is {2}".format(a, b, sum))

.. admonition:: Adding two numbers given as arguments (sum-2args.py)
    :class: dropdown
   
        .. code-block:: python

            import sys
            
            x = int(sys.argv[1])
            y = int(sys.argv[2])
            
            sum = x + y
            
            print("The sum of the two numbers is: {0}".format(sum))

Now for the examples: 

.. admonition:: Example, Running a Python script in the allocation we made further up. Notice that since we asked for 4 cores, the script is run 4 times, since it is a serial script
    :class: dropdown
   
        .. code-block:: sh

            b-an01 [~]$ srun python sum-2args.py 3 4
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            The sum of the two numbers is: 7
            b-an01 [~]$             
            
.. admonition:: Example, Running a Python script in the above allocation, but this time a script that expects input from you.
    :class: dropdown
   
        .. code-block:: sh            
            
            b-an01 [~]$ srun python add2.py 
            2
            3
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5
            Enter the first number: Enter the second number: The sum of 2 and 3 is 5

Batch mode
##########

Serial code
'''''''''''

.. admonition:: Running on Kebnekaise, SciPy-bundle/2021.05 and Python/3.9.5, serial code 
    :class: dropdown
   
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
            
            # Run your Python script 
            python <my_program.py>
            
            
Serial code + self-installed package in virt. env. 
''''''''''''''''''''''''''''''''''''''''''''''''''

.. admonition:: Running on Kebnekaise, SciPy-bundle/2021.05, Python/3.9.5 + Python package you have installed yourself with virtual environment. Serial code
    :class: dropdown
   
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5 and compatible SciPy-bundle
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 SciPy-bundle/2021.05
            
            # Activate your virtual environment. Note that you either need to have added the location to your path, or give the full path
            source <path-to-virt-env>/bin/activate
 
            # Run your Python script 
            python <my_program.py>

GPU code
'''''''' 

.. admonition:: Running on Kebnekaise, SciPy-bundle/2021.05, Python/3.9.5 + TensorFlow/2.6.0-CUDA-11.3.1, GPU code
    :class: dropdown
   
        .. code-block:: sh

            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:10:00 # Asking for 10 minutes
            # Asking for one K80 card
            #SBATCH --gres=gpu:k80:1
            
            # Load any modules you need 
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5 TensorFlow/2.6.0-CUDA-11.3.1
          
            # Run your Python script 
            python <my_tf_program.py>
            

The recommended TensorFlow version for this course is 2.6.0. The module is compatible with Python 3.9.5 (automatically loaded when you load TensorFlow and its other prerequisites).            




Exercises
---------

.. challenge:: Run the first serial example from further up on the page for this short Python code (sum-2args.py)
    
    .. code-block:: python
    
        import sys
            
        x = int(sys.argv[1])
        y = int(sys.argv[2])
            
        sum = x + y
            
        print("The sum of the two numbers is: {0}".format(sum))
        
    Remember to give the two arguments to the program in the batch script.

.. solution::
    :class: dropdown
    
          This is for Kebnekaise. Adding the numbers 2 and 3. 
          
          .. code-block:: sh
 
            #!/bin/bash
            #SBATCH -A SNIC2022-22-641 # Change to your own after the course
            #SBATCH --time=00:05:00 # Asking for 5 minutes
            #SBATCH -n 1 # Asking for 1 core
            
            # Load any modules you need, here for Python 3.9.5
            module load GCC/10.3.0  OpenMPI/4.1.1 Python/3.9.5
            
            # Run your Python script 
            python sum-2args.py 2 3 
