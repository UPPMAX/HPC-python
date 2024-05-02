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


   
   While Python does not run fast, it is still well suited for machine learning. For instance, it is fairly easy to code in, and this is particularly useful in machine learning where the right solution is rarely known from the start. A lot of tests and experimentation is needed, and the program usually goes through many iterations. In addition, there are a lot of useful libraries written for machine learning in Python, making it a good choice for this area. 

Some of the most used libraries in Python for machine learning are: 

- PyTorch
- scikit-learn
- TensorFlow

These are all available at UPPMAX and HPC2N. 

In this course we will look at two examples: PyTorch and TensorFlow, and show how you run them at our centres. 

There are some examples for beginners at https://machinelearningmastery.com/start-here/#python and at https://pytorch.org/tutorials/beginner/pytorch_with_examples.html 

Pandas and matplotlib
---------------------

This is the same example that was shown in the section about loading and running Python, but now changed slightly to run as a batch job. The main difference is that here we cannot open the plot directly, but have to save to a file instead. You can see the change inside the Python script. 

.. tabs::

   .. tab:: Directly

      Remove the # if running on Kebnekaise

      .. code-block::

         import pandas as pd
         #import matplotlib
         import matplotlib.pyplot as plt

         #matplotlib.use('TkAgg')

         dataframe = pd.read_csv("scottish_hills.csv")
         x = dataframe.Height
         y = dataframe.Latitude
         plt.scatter(x, y)
         plt.show()

   .. tab:: From a Batch-job 

      Remove the # if running on Kebnekaise. The script below can be found as ``pandas_matplotlib-batch.py`` or ``pandas_matplotlib-batch-kebnekaise.py`` in the ``Exercises/examples/programs`` directory. 

      .. code-block:: 

         import pandas as pd
         #import matplotlib
         import matplotlib.pyplot as plt
         
         #matplotlib.use('TkAgg')

         dataframe = pd.read_csv("scottish_hills.csv")
         x = dataframe.Height
         y = dataframe.Latitude
         plt.scatter(x, y)
         plt.savefig("myplot.png")

.. hint::

   Type along!
   
Batch scripts for running on Rackham and Kebnekaise. 

.. tabs:: 

   .. tab:: Rackham 

      .. code-block::

         #!/bin/bash -l
         #SBATCH -A naiss2024-22-415
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH -n 1 # Asking for 1 core

         # Load any modules you need, here for Python 3.11.8
         ml python/3.11.8

         # Run your Python script
         python pandas_matplotlib-batch.py 

   .. tab:: Kebnekaise 

      .. code-block::

         #!/bin/bash
         #SBATCH -A hpc2n2024-052
         #SBATCH --time=00:05:00 # Asking for 5 minutes
         #SBATCH -n 1 # Asking for 1 core

         # Load any modules you need, here for Python 3.11.3
         ml GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

         # Run your Python script
         python pandas_matplotlib-batch-kebnekaise.py

Submit with ``sbatch <batch-script.sh>``. 

The batch scripts can be found in the directories for hpc2n and uppmax, under ``Exercises/examples/``, and they are named ``pandas_matplotlib-batch.sh`` and ``pandas_matplotlib-batch-kebnekaise.sh``. 

PyTorch
-------

PyTorch has: 

- An n-dimensional Tensor, similar to numpy, but can run on GPUs
- Automatic differentiation for building and training neural networks

The example we will use in this course is taken from the official PyTorch page: https://pytorch.org/ and the problem is of fitting :math:`y=sin⁡(x)` with a third order polynomial. We will run an example as a batch job. 

.. admonition:: We use PyTorch Tensors to fit a third order polynomial to a sine function. The forward and backward passes through the network are manually implemented. 
    :class: dropdown

        The below program can be found in the ``Exercises/examples/programs`` directory under the name ``pytorch_fitting_gpu.py``. 

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

.. hint::

   Type along!

In order to run this at HPC2N/UPPMAX you should either do a batch job or run interactively on compute nodes. Remember, you should not run long/resource heavy jobs on the login nodes, and they also do not have GPUs if you want to use that.  

This is an example of a batch script for running the above example, using PyTorch 2.1.x and Python 3.11.x, and running on GPUs. 

.. admonition:: Example batch script, running on Kebnekaise 
    :class: dropdown

        .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A hpc2n2024-052
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # The following two lines splits the output in a file for any errors and a file for other output. 
            #SBATCH --error=job.%J.err
            #SBATCH --output=job.%J.out
            # Asking for one V100
            #SBATCH --gres=gpu:V100:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 
            
            srun python pytorch_fitting_gpu.py
            

.. admonition:: UPPMAX as run in an interactive Snowy session
    :class: dropdown

        .. code-block:: sh

            $ interactive -A naiss2024-22-415 -n 1 -M snowy --gres=gpu:1  -t 1:00:01 
            You receive the high interactive priority.

            Please, use no more than 8 GB of RAM.

            Waiting for job 6907137 to start...
            Starting job now -- you waited for 90 seconds.

            $  ml uppmax
            $  ml python/3.11.8
            $  module load python_ML_packages/3.11.8-gpu
            $  cd /proj/naiss2024-22-415/<user-dir>/HPC-python/Exercises/examples/programs
            $ srun python pytorch_fitting_gpu.py
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

We are using Tensorflow 2.11.0-CUDA-11.7.0 (and Python 3.10.4) at HPC2N, since that is the newest GPU-enabled TensorFlow currently installed there. 

On UPPMAX we are using TensorFlow 2.15.0 (included in python_ML_packages/3.11.8-gpu) and Python 3.11.8. 

.. hint::

   Type along!

.. tabs::
  
   .. tab:: HPC2N

      Since we need scikit-learn, we are also loading the scikit-learn/1.1.2 which is compatible with the other modules we are using.  

      Thus, load modules: ``GCC/11.3.0  OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2`` in your batch script.  
      
   .. tab:: UPPMAX
   
      UPPMAX has scikit-learn in the python_ML_packages, so we do not need to load anything extra there. 

        - Load modules: ``module load uppmax python/3.11.8 python_ML_packages/3.11.8-gpu``
           - On Rackham we should use python_ML-packages/3.11.8-cpu, while on a GPU node the GPU version should be loaded (like we do in this example, which will work either in a batch script submitted to Snowy or in an interactive job running on Snowy). 

  

.. admonition:: We will work with this example (example-tf.py) 
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

      Example batch script for Kebnekaise, TensorFlow version 2.11.0 and Python version 3.10.4, and scikit-learn 1.1.2 
      
      .. code-block:: sh 
        
            #!/bin/bash 
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A hpc2n2024-052
            # We are asking for 5 minutes
            #SBATCH --time=00:05:00
            # Asking for one V100
            #SBATCH --gres=gpu:v100:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load GCC/11.3.0 Python/3.10.4 OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2 
            
            # Run your Python script 
            python example-tf.sh 
            
   .. tab:: UPPMAX

      Example batch script for Snowy, Python version 3.11.8, and the python_ML_packages/3.11.8-gpu containing Tensorflow 
      
      .. code-block:: sh 
        
            #!/bin/bash -l  
            # Remember to change this to your own project ID after the course! 
            #SBATCH -A naiss2024-22-415
            # We want to run on Snowy
            #SBATCH -M snowy
            # We are asking for 15 minutes
            #SBATCH --time=00:15:00
            #SBATCH --gres=gpu:1
            
            # Remove any loaded modules and load the ones we need
            module purge  > /dev/null 2>&1
            module load uppmax
            module load python_ML_packages/3.11.8-gpu 
            
            # Run your Python script 
            python example-tf.py 
            
            
Submit with ``sbatch example-tf.sh``. After submitting you will (as usual) be given the job-id for your job. You can check on the progress of your job with ``squeue -u <username>`` or ``scontrol show <job-id>``. 

Note: if you are logged in to Rackham on UPPMAX and have submitted a GPU job to Snowy, then you need to use this to see the job queue: 

``squeue -M snowy -u <username>``


General
-------

You almost always want to run several iterations of your machine learning code with changed parameters and/or added layers. If you are doing this in a batch job, it is easiest to either make a batch script that submits several variations of your Python script (changed parameters, changed layers), or make a script that loops over and submits jobs with the changes. 

Running several jobs from within one job
########################################

.. hint:: 

   Do NOT type along!

This example shows how you would run several programs or variations of programs sequentially within the same job: 

.. tabs::

   .. tab:: HPC2N

      Example batch script for Kebnekaise, TensorFlow version 2.11.0 and Python version 3.11.3

      .. code-block:: sh 
        
         #!/bin/bash 
         # Remember to change this to your own project ID after the course! 
         #SBATCH -A hpc2n2024-052
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

      Example batch script for Snowy, TensorFlow version 2.15 and Python version 3.11.8. 
      
      .. code-block:: sh 

         #!/bin/bash -l
         # Remember to change this to your own project ID after the course!
         #SBATCH -A naiss2024-22-415
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
         module load python_ML_packages/3.11.8-gpu
         # Output to file - not needed if your job creates output in a file directly
         # In this example I also copy the output somewhere else and then run another executable (or you could just run the same executable for different parameters).
         python tf_program.py 1 2 > myoutput1 2>&1
         cp myoutput1 mydatadir
         python tf_program.py 3 4 > myoutput2 2>&1
         cp myoutput2 mydatadir
         python tf_program.py 5 6 > myoutput3 2>&1
         cp myoutput3 mydatadir

Exercises
---------

.. challenge::

   Try to modify the files ``pandas_matplotlib-linreg-<rackham/kebnekaise>.py`` and ``pandas_matplotlib-linreg-pretty-<rackham/kebnekaise>.py so they could be run from a batch job (change the pop-up plots to save-to-file).

   Also change the batch script ``pandas_matplotlib.sh`` (or ``pandas_matplotlib-kebnekaise.sh``) to run your modified python codes. 

.. challenge:: 

   In this exercise you will be using the course environment that you prepared in the "Install packages" section (here: https://uppmax.github.io/HPC-python/install_packages.html#prepare-the-course-environment). 

   You will run the Python code ``simple_lightgbm.py`` found in the ``Exercises/examples/programs`` directory. The code was taken from https://github.com/microsoft/LightGBM/tree/master and lightly modified. 

   Try to write a batch script that runs this code. Remember to activate the course environment. 

   .. tabs::

      .. tab:: simple_lightgbm.py 

         .. code-block::
         
            # coding: utf-8
            from pathlib import Path
  
            import pandas as pd
            from sklearn.metrics import mean_squared_error

            import lightgbm as lgb

            print("Loading data...")
            # load or create your dataset
            df_train = pd.read_csv(str("regression.train"), header=None, sep="\t")
            df_test = pd.read_csv(str("regression.test"), header=None, sep="\t")

            y_train = df_train[0]
            y_test = df_test[0]
            X_train = df_train.drop(0, axis=1)
            X_test = df_test.drop(0, axis=1)

            # create dataset for lightgbm
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

            # specify your configurations as a dict
            params = {
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": {"l2", "l1"},
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": 0,
            }

            print("Starting training...")
            # train
            gbm = lgb.train(
                params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, callbacks=[lgb.early_stopping(stopping_rounds=5)]
            )

            print("Saving model...")
            # save model to file
            gbm.save_model("model.txt")
 
            print("Starting predicting...")
            # predict
            y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            # eval
            rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
            print(f"The RMSE of prediction is: {rmse_test}")

      .. tab:: Rackham 

         .. admonition:: Click to reveal the solution! 
             :class: dropdown

                   .. code-block:: 

                      #!/bin/bash -l
                      # Change to your own project ID after the course!
                      #SBATCH -A naiss2024-22-415
                      # We are asking for 10 minutes
                      #SBATCH --time=00:10:00
                      #SBATCH -n 1

                      # Set a path where the example programs are installed. 
                      # Change the below to your own path to where you placed the example programs
                      MYPATH=/proj/hpc-python/<mydir-name>/HPC-python/Exercises/examples/programs/
                      # Activate the course environment (assuming it was called vpyenv) 
                      source /proj/hpc-python/<mydir-name>/<path-to-my-venv>/vpyenv/bin/activate
                      # Remove any loaded modules and load the ones we need
                      module purge  > /dev/null 2>&1
                      module load uppmax
                      module load python/3.11.8

                      # Run your Python script
                      python $MYPATH/simple_lightgbm.py
                                     
      .. tab:: Kebnekaise 

         .. admonition:: Click to reveal the solution! 
             :class: dropdown 

                   .. code-block:: 

                      #!/bin/bash
                      # Change to your own project ID after the course!
                      #SBATCH -A hpc2n2024-052
                      # We are asking for 10 minutes
                      #SBATCH --time=00:10:00
                      #SBATCH -n 1

                      # Set a path where the example programs are installed. 
                      # Change the below to your own path to where you placed the example programs
                      MYPATH=/proj/nobackup/python-hpc/<mydir-name>/HPC-python/Exercises/examples/programs/

                      # Remove any loaded modules and load the ones we need
                      module purge  > /dev/null 2>&1
                      module load GCC/12.3.0 Python/3.11.3 SciPy-bundle/2023.07 matplotlib/3.7.2

                      # Activate the course environment (assuming it was called vpyenv) 
                      source /proj/nobackup/python-hpc/<mydir-name>/<path-to-my-venv>/vpyenv/bin/activate

                      # Run your Python script
                      python $MYPATH/simple_lightgbm.py


.. keypoints::

  - At all clusters you will find PyTorch, TensorFlow, Scikit-learn
  - The loading are slightly different at the clusters
     - UPPMAX: All these tools are available from the modules ``ml python_ML_packages/3.11.8 python/3.11.8``
     - HPC2N: 
       - For TensorFlow: ``ml GCC/11.3.0  OpenMPI/4.1.4 TensorFlow/2.11.0-CUDA-11.7.0 scikit-learn/1.1.2`` 
       - For the rest: ``ml GCC/12.3.0 OpenMPI/4.1.5 SciPy-bundle/2023.07 matplotlib/3.7.2 PyTorch/2.1.2 scikit-learn/1.3.1``

