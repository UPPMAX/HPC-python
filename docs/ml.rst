Using Python for Machine Learning jobs
======================================

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

In order to run this at HPC2N (and at UPPMAX?) you should use a batch job. 

This is an example of a batch script for running the above example, using PyTorch 1.10.0 and Python 3.9.5, running on GPUs. 

.. admonition:: Example batch script, running the above example (assuming it is named pytorch_fitting_gpu.py) 
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

.. admonition:: Example batch script, TensorFlow version 2.6.0 and Python version 3.9.5, and the scikit-learn we installed 
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
-------

You almost always want to run several iterations of your machine learning code with changed parameters and/or added layers. If you are doing this in a batch job, it is easiest to either make a batch script that submits several variations of your Python script (changed parameters, changed layers), or make a script that loops over and submits jobs with the changes. 

Running several jobs from within one job
''''''''''''''''''''''''''''''''''''''''

This example shows how you would run several programs or variations of programs sequentially within the same job: 

.. admonition:: Example batch script, TensorFlow version 2.6.0 and Python version 3.9.5) 
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


