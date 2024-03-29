Using Python for Machine Learning jobs
======================================

.. questions::

   - Which machine learning tools are installed at HPC2N and UPPMAX?
   - How to start the tools at HPC2N and UPPMAX
   - How to deploy GPU:s at HPC2N and UPPMAX?
   

.. objectives::

   - Get general overview of installed Machine Learning tools at HPC2N and UPPMAX
   - Get started with Machine learning in Python
   - Code along and demos (Kebnekaise and Snowy)


   
   While Python does not run fast, it is still well suited for machine learning. However, it is fairly easy to code in, and this is particularly useful in machine learning where the right solution is rarely known from the start. A lot of tests and experimentation is needed, and the program usually goes through many iterations. In addition, there are a lot of useful libraries written for machine learning in Python, making it a good choice for this area. 

Some of the most used libraries in Python for machine learning are: 

- PyTorch
- scikit-learn
- TensorFlow

These are all available at UPPMAX and HPC2N. 

In this course we will look at two examples: PyTorch and TensorFlow, and show how you run them at our centres. 

There are some examples for beginners at https://machinelearningmastery.com/start-here/#python and at https://pytorch.org/tutorials/beginner/pytorch_with_examples.html 

PyTorch
-------

PyTorch has: 

- An n-dimensional Tensor, similar to numpy, but can run on GPUs
- Automatic differentiation for building and training neural networks

The example we will use in this course is taken from the official PyTorch page: https://pytorch.org/ and the problem is of fitting :math:`y=sin⁡(x)` with a third order polynomial. We will run an example as a batch job. 

.. admonition:: We use PyTorch Tensors to fit a third order polynomial to a sine function. The forward and backward passes through the network are manually implemented. 
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

You can find the full list of examples for this problem here: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

In order to run this at HPC2N/UPPMAX you should either do a batch job or run interactively on compute nodes. Remember, you should not run long/resource heavy jobs on the login nodes, and they also do not have GPUs if you want to use that.  

This is an example of a batch script for running the above example, using PyTorch 1.10.0 and Python 3.9.5, running on GPUs. 

.. admonition:: Example batch script, running the above example on Kebnekaise (assuming it is named pytorch_fitting_gpu.py) 
    :class: dropdown

        .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID! 
            #SBATCH -A hpc2nXXXX-YYY
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # The following two lines splits the output in a file for any errors and a file for other output. 
            #SBATCH --error=job.%J.err
            #SBATCH --output=job.%J.out
            # Asking for one V100
            #SBATCH --gres=gpu:V100:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0  OpenMPI/4.1.1 PyTorch/1.10.0-CUDA-11.3.1
            
            srun python pytorch_fitting_gpu.py
            

.. admonition:: UPPMAX as a run in an interactive Snowy session
    :class: dropdown

        .. code-block:: sh

            [bjornc@rackham3 ~]$ interactive -A naiss2023-22-1126 -n 1 -M snowy --gres=gpu:1  -t 1:00:01 
            You receive the high interactive priority.

            Please, use no more than 8 GB of RAM.

            Waiting for job 6907137 to start...
            Starting job now -- you waited for 90 seconds.

            [bjornc@s160 ~]$  ml uppmax
            [bjornc@s160 ~]$  ml python/3.9.5
            [bjornc@s160 ~]$  module load python_ML_packages/3.9.5-gpu
            [bjornc@s160 ~]$  cd /proj/naiss2023-22-1126/bjornc/HPC-python/Exercises/examples/programs
            [bjornc@s160 programs]$ srun python pytorch_fitting_gpu.py
            99 134.71942138671875
            199 97.72868347167969
            299 71.6167221069336
            399 53.178802490234375
            499 40.15779113769531
            599 30.9610652923584
            699 24.464630126953125
            799 19.875120162963867
            899 16.632421493530273
            999 14.341087341308594
            1099 12.721846580505371
            1199 11.577451705932617
            1299 10.76859188079834
            1399 10.196844100952148
            1499 9.792669296264648
            1599 9.506935119628906
            1699 9.304922103881836
            1799 9.162087440490723
            1899 9.061092376708984
            1999 8.989676475524902
            Result: y = 0.013841948471963406 + 0.855550229549408 x + -0.002387965563684702 x^2 + -0.09316103905439377 x^3



TensorFlow
----------

The example comes from https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/ but there are also good examples at https://www.tensorflow.org/tutorials 

We are using Tensorflow 2.6.0-CUDA-11.3-1 (and Python 3.9.5) at HPC2N (also 3.9.5 at UPPMAX). 

.. tabs::
  
   .. tab:: HPC2N

      Since there is no scikit-learn for these versions, we have to install that too: 

      Installing scikit-learn compatible with TensorFlow version 2.7.1 and Python version 3.10.4 

      
        - Load modules: ``module load GCC/11.2.0 OpenMPI/4.1.1 SciPy-bundle/2021.10 TensorFlow/2.7.1``
        - Activate the virtual environment we created earlier: ``source <path-to-install-dir>/vpyenv/bin/activate``
        - ``pip install --no-cache-dir --no-build-isolation scikit-learn``
        
      We can now use scikit-learn in our example. 
      
   .. tab:: UPPMAX
   
      UPPMAX has scikit-learn in the scikit-learn/0.22.1 module. We also need the python_ML module for Tensorflow, so let's just load those

        - Load modules: ``module load python_ML_packages/3.9.5-gpu python/3.9.5``
           - On Rackham we should use python_ML-packages/3.9.5-cpu, while on a GPU node the GPU version should be loaded 

      

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

.. tabs::

   .. tab:: HPC2N

      Example batch script for Kebnekaise, TensorFlow version 2.6.0 and Python version 3.9.5, and the scikit-learn we installed 
      
      .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A hpc2nXXXX-YYY
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # Asking for one V100
            #SBATCH --gres=gpu:v100:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/10.3.0 OpenMPI/4.1.1 SciPy-bundle/2021.05 TensorFlow/2.6.0-CUDA-11.3.1
            
            # Activate the virtual environment we installed to 
            source <path-to-install-dir>/vpyenv/bin/activate 
            
            # Run your Python script 
            python <my_tf_program.py> 
            
   .. tab:: UPPMAX

      Example batch script for Snowy, Python version 3.9.5, and the python_ML_packages containing Tensorflow 
      
      .. code-block:: sh 
        
            #!/bin/bash -l  
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A naiss2023-22-1126
            # We want to run on Snowy
            #SBATCH -M snowy
            # We are asking for 15 minutes
            #SBATCH --time=00:15:00
            #SBATCH --gres=gpu:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load uppmax
            module load python_ML_packages/3.9.5-gpu 
            module load python/3.9.5 # to get some extra packages

            
            # Run your Python script 
            python <my_tf_program.py> 
            
            
Submit with ``sbatch <myjobscript.sh>``. After submitting you will (as usual) be given the job-id for your job. You can check on the progress of your job with ``squeue -u <username>`` or ``scontrol show <job-id>``. 

Note: if you are logged in to Rackham on UPPMAX and have submitted a GPU job to Snowy, then you need to use this to see the job queue: 

``squeue -M snowy -u <username>``


General
-------

You almost always want to run several iterations of your machine learning code with changed parameters and/or added layers. If you are doing this in a batch job, it is easiest to either make a batch script that submits several variations of your Python script (changed parameters, changed layers), or make a script that loops over and submits jobs with the changes. 

Running several jobs from within one job
########################################

This example shows how you would run several programs or variations of programs sequentially within the same job: 

.. tabs::

   .. tab:: HPC2N

      Example batch script for Kebnekaise, TensorFlow version 2.6.0 and Python version 3.9.5

      .. code-block:: sh 
        
         #!/bin/bash 
         # Remember to change this to your own project ID after the course! 
         #SBATCH -A hpc2nXXXX-YYY
         # We are asking for 5 minutes
         #SBATCH --time=00:05:00
         # Asking for one V100 
         #SBATCH --gres=gpu:v100:1
         # Remove any loaded modules and load the ones we need
         module purge  > /dev/null 2>&1
         module load GCC/10.3.0 OpenMPI/4.1.1 SciPy-bundle/2021.05 TensorFlow/2.6.0-CUDA-11.3-1 
         # Output to file - not needed if your job creates output in a file directly 
         # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters). 
         python <my_tf_program.py> <param1> <param2> > myoutput1 2>&1
         cp myoutput1 mydatadir
         python <my_tf_program.py> <param3> <param4> > myoutput2 2>&1
         cp myoutput2 mydatadir
         python <my_tf_program.py> <param5> <param6> > myoutput3 2>&1
         cp myoutput3 mydatadir

   .. tab:: UPPMAX

      Example batch script for Snowy, TensorFlow version 2.5.0 and Python version 3.9.5.
      
      .. code-block:: sh 

         #!/bin/bash -l
         # Remember to change this to your own project ID after the course!
         #SBATCH -A naiss2023-22-1126
         # We are asking for at least 1 hour
         #SBATCH --time=01:00:01
         #SBATCH -M snowy
         #SBATCH --gres=gpu:1
         #SBATCH --mail-type=begin        # send email when job begins
         #SBATCH --mail-type=end          # send email when job ends
         #SBATCH --mail-user=bjorn.claremar@uppmax.uu.se
         # Remove any loaded modules and load the ones we need
         module purge  > /dev/null 2>&1
         module load uppmax
         module load python_ML_packages/3.9.5-gpu
         module load python/3.9.5 # to get some extra packages
         module load TensorFlow/2.5.0-fosscuda-2020b
         # Output to file - not needed if your job creates output in a file directly
         # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).
         python tf_program.py 1 2 > myoutput1 2>&1
         cp myoutput1 mydatadir
         python tf_program.py 3 4 > myoutput2 2>&1
         cp myoutput2 mydatadir
         python tf_program.py 5 6 > myoutput3 2>&1
         cp myoutput3 mydatadir


.. keypoints::

  - At all clusters you will find PyTorch, TensorFlow, Scikit-learn
  - The loading are slightly different at the clusters
     - UPPMAX: All tools are available from the modules ``ml python_ML_packages python/3.9.5``
     - HPC2N: ``module load GCC/11.3.0 OpenMPI/4.1.1 SciPy-bundle/2021.05 TensorFlow/2.6.0-CUDA-11.3.1``


