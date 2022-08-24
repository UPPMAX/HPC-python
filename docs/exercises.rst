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

Machine Learning
################

. admonition:: We use PyTorch Tensors to fit a third order polynomial to a sine function. The forward and backward passes through the network are manually implemented. 
    :class: dropdown

        .. code-block:: python
        
            # -*- coding: utf-8 -*-
            
            import torch
            import math
            
            dtype = torch.float
            device = torch.device("cpu")
            # device = torch.device("cuda:0") # Uncomment this to run on GPU
            
            # Create random input and output data
            x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
            y = torch.sin(x)
            
            # Randomly initialize weights
            a = torch.randn((), device=device, dtype=dtype)
            b = torch.randn((), device=device, dtype=dtype)
            c = torch.randn((), device=device, dtype=dtype)
            d = torch.randn((), device=device, dtype=dtype)
            
            learning_rate = 1e-6
            for t in range(2000):
                # Forward pass: compute predicted y
                y_pred = a + b * x + c * x ** 2 + d * x ** 3
                
                # Compute and print loss
                loss = (y_pred - y).pow(2).sum().item()
                if t % 100 == 99:
                    print(t, loss)
                
                # Backprop to compute gradients of a, b, c, d with respect to loss
                grad_y_pred = 2.0 * (y_pred - y)
                grad_a = grad_y_pred.sum()
                grad_b = (grad_y_pred * x).sum()
                grad_c = (grad_y_pred * x ** 2).sum()
                grad_d = (grad_y_pred * x ** 3).sum()
                
                # Update weights using gradient descent
                a -= learning_rate * grad_a
                b -= learning_rate * grad_b
                c -= learning_rate * grad_c
                d -= learning_rate * grad_d
                
            print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

This is an example of a batch script for running the above example, using PyTorch 1.10.0 and Python 3.9.5, running on GPUs. 

.. admonition:: Example batch script, running the above example on Kebnekaise (assuming it is named pytorch_fitting_gpu.py) 
    :class: dropdown

        .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A SNIC2022-22-641
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # The following two lines splits the output in a file for any errors and a file for other output. 
            #SBATCH --error=job.%J.err
            #SBATCH --output=job.%J.out
            # Asking for one K80
            #SBATCH --gres=gpu:k80:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0  OpenMPI/4.1.1 PyTorch/1.10.0-CUDA-11.3.1
            
            srun python pytorch_fitting_gpu.py

TensorFlow
----------

The example comes from https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/ but there are also good examples at https://www.tensorflow.org/tutorials 

We are using Tensorflow 2.6.0 and Python 3.9.5. Since there is no scikit-learn for these versions, we have to install that too: 

.. admonition:: Installing scikit-learn compatible with TensorFlow version 2.6.0 and Python version 3.9.5 
    :class: dropdown
      
        - Load modules: ``module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1``
        - Create virtual environment: ``virtualenv --system-site-packages <path-to-install-dir>/vpyenv``
        - Activate the virtual environment: ``source <path-to-install-dir>/vpyenv/bin/activate``
        - ``pip install --no-cache-dir --no-build-isolation scikit-learn``
        
We can now use scikit-learn in our example. 

.. admonition:: We will work with this example  
    :class: dropdown

        .. code-block:: sh 
        
            # mlp for binary classification
            from pandas import read_csv
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense
            # load the dataset
            path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
            df = read_csv(path, header=None)
            # split into input and output columns
            X, y = df.values[:, :-1], df.values[:, -1]
            # ensure all data are floating point values
            X = X.astype('float32')
            # encode strings to integer
            y = LabelEncoder().fit_transform(y)
            # split into train and test datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            # determine the number of input features
            n_features = X_train.shape[1]
            # define model
            model = Sequential()
            model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
            model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
            model.add(Dense(1, activation='sigmoid'))
            # compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # fit the model
            model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
            # evaluate the model
            loss, acc = model.evaluate(X_test, y_test, verbose=0)
            print('Test Accuracy: %.3f' % acc)
            # make a prediction
            row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
            yhat = model.predict([row])
            print('Predicted: %.3f' % yhat)


In order to run the above example, we will create a batch script and submit it.             

.. admonition:: Example batch script for Kebnekaise, TensorFlow version 2.6.0 and Python version 3.9.5, and the scikit-learn we installed 
    :class: dropdown

        .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A SNIC2022-22-641
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # Asking for one K80 
            #SBATCH --gres=gpu:k80:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1
            
            # Activate the virtual environment we installed to 
            source <path-to-install-dir>/vpyenv/bin/activate 
            
            # Run your Python script 
            python <my_tf_program.py> 
            
            
Submit with ``sbatch <myjobscript.sh>``. After submitting you will (as usual) be given the job-id for your job. You can check on the progress of your job with ``squeue -u <username>`` or ``scontrol show <job-id>``. 

The output and errors will in this case be written to ``slurm-<job-id>.out``. 

General
#######

You almost always want to run several iterations of your machine learning code with changed parameters and/or added layers. If you are doing this in a batch job, it is easiest to either make a batch script that submits several variations of your Python script (changed parameters, changed layers), or make a script that loops over and submits jobs with the changes. 

Running several jobs from within one job
''''''''''''''''''''''''''''''''''''''''

This example shows how you would run several programs or variations of programs sequentially within the same job: 

.. admonition:: Example batch script for Kebnekaise, TensorFlow version 2.6.0 and Python version 3.9.5) 
    :class: dropdown

        .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A SNIC2022-22-641
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # Asking for one K80 
            #SBATCH --gres=gpu:k80:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1
            
            # Output to file - not needed if your job creates output in a file directly 
            # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).
            
            python <my_tf_program.py> <param1> <param2> > myoutput1 2>&1
            cp myoutput1 mydatadir
            python <my_tf_program.py> <param3> <param4> > myoutput2 2>&1
            cp myoutput2 mydatadir
            python <my_tf_program.py> <param5> <param6> > myoutput3 2>&1
            cp myoutput3 mydatadir



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
