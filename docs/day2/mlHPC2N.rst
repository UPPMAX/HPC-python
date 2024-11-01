Using Python for Machine Learning jobs 2
================================================================

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
-------

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

Horovod
-------

As the training is one of the most computationally demanding steps in a ML workflow,
it would be worth it to optimize this step. Horovod is a framework dedicated to
make more efficient the training step by distributing the workload across several
nodes, each consisting of some CPUs and GPUs. An example on the usage of Horovod
can be found in the course `Upscaling AI workflows <https://enccs.github.io/upscalingAI/hvd_intro/>`_
offered by ENCCS.


   .. admonition:: ``Transfer_Learning_NLP_Horovod.py``
      :class: dropdown

      .. code-block:: python

         import numpy as np
         import pandas as pd
         import time
         import tensorflow as tf
         
         import tempfile
         import pathlib
         import shutil
         import tempfile
         import os
         import argparse
         
         # Suppress tensorflow logging outputs
         # os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
         
         import tensorflow_hub as hub
         from sklearn.model_selection import train_test_split
         
         logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
         shutil.rmtree(logdir, ignore_errors=True)
         
         # Parse input arguments
         
         parser = argparse.ArgumentParser(description='Transfer Learning Example',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
         
         parser.add_argument('--log-dir', default=logdir,
                             help='tensorboard log directory')
         
         parser.add_argument('--num-worker', default=1,
                             help='number of workers for training part')
         
         parser.add_argument('--batch-size', type=int, default=32,
                             help='input batch size for training')
         
         parser.add_argument('--base-lr', type=float, default=0.01,
                             help='learning rate for a single GPU')
         
         parser.add_argument('--epochs', type=int, default=40,
                             help='number of epochs to train')
         
         parser.add_argument('--momentum', type=float, default=0.9,
                             help='SGD momentum')
         
         parser.add_argument('--target-accuracy', type=float, default=.96,
                             help='Target accuracy to stop training')
         
         parser.add_argument('--patience', type=float, default=2,
                             help='Number of epochs that meet target before stopping')
         
         parser.add_argument('--use-checkpointing', default=False, action='store_true')
         
         # Step 10: register `--warmup-epochs`
         parser.add_argument('--warmup-epochs', type=float, default=5,
                             help='number of warmup epochs')
         
         args = parser.parse_args()
         
         # Define a function for a simple learning rate decay over time
         
         def lr_schedule(epoch):
             
             if epoch < 15:
                 return args.base_lr
             if epoch < 25:
                 return 1e-1 * args.base_lr
             if epoch < 35:
                 return 1e-2 * args.base_lr
             return 1e-3 * args.base_lr
         
         ##### Steps
         # Step 1: import Horovod
         import horovod.tensorflow.keras as hvd
         
         hvd.init()
         
         # Nomrally Step 2: pin to a GPU
         gpus = tf.config.list_physical_devices('GPU')
         for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
         if gpus:
             tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
         
         # Step 2: but in our case
         # gpus = tf.config.list_physical_devices('GPU')
         # if gpus:
         #    tf.config.experimental.set_memory_growth(gpus[0], True)
         
         # Step 3: only set `verbose` to `1` if this is the root worker.
         if hvd.rank() == 0:
             print("Version: ", tf.__version__)
             print("Hub version: ", hub.__version__)
             print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
             print('Number of GPUs :',len(tf.config.list_physical_devices('GPU')))
             verbose = 1
         else:
             verbose = 0
         #####
         
         if os.path.exists('dataset.pkl'):
             df = pd.read_pickle('dataset.pkl')
         else:
             df = pd.read_csv('https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip', 
                      compression='zip', low_memory=False)
             df.to_pickle('dataset.pkl')
         
         train_df, remaining = train_test_split(df, random_state=42, train_size=0.9, stratify=df.target.values)
         valid_df, _  = train_test_split(remaining, random_state=42, train_size=0.09, stratify=remaining.target.values)
         
         if hvd.rank() == 0:
             print("The shape of training {} and validation {} datasets.".format(train_df.shape, valid_df.shape))
             print("##-------------------------##")
         
         buffer_size = train_df.size
         #train_dataset = tf.data.Dataset.from_tensor_slices((train_df.question_text.values, train_df.target.values)).repeat(args.epochs*2).shuffle(buffer_size).batch(args.batch_size)
         #valid_dataset = tf.data.Dataset.from_tensor_slices((valid_df.question_text.values, valid_df.target.values)).repeat(args.epochs*2).batch(args.batch_size)
         
         train_dataset = tf.data.Dataset.from_tensor_slices((train_df.question_text.values, train_df.target.values)).repeat().shuffle(buffer_size).batch(args.batch_size)
         valid_dataset = tf.data.Dataset.from_tensor_slices((valid_df.question_text.values, valid_df.target.values)).repeat().batch(args.batch_size)
         
         module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
         embeding_size = 128
         name_of_model = 'nnlm-en-dim128'
         
         def create_model(module_url, embed_size, name, trainable=False):
             hub_layer = hub.KerasLayer(module_url, input_shape=[], output_shape=[embed_size], dtype = tf.string, trainable=trainable)
             model = tf.keras.models.Sequential([hub_layer,
                                                 tf.keras.layers.Dense(256, activation='relu'),
                                                 tf.keras.layers.Dense(64, activation='relu'),
                                                 tf.keras.layers.Dense(1, activation='sigmoid')])
             
             # Step 9: Scale the learning rate by the number of workers.
             opt = tf.optimizers.SGD(learning_rate=args.base_lr * hvd.size(), momentum=args.momentum)
             # opt = tf.optimizers.Adam(learning_rate=args.base_lr * hvd.size())
         
             #Step 4: Wrap the optimizer in a Horovod distributed optimizer
             opt = hvd.DistributedOptimizer(opt,
                                            backward_passes_per_step=1, 
                                            average_aggregated_gradients=True
                                            )
         
             # For Horovod: We specify `experimental_run_tf_function=False` to ensure TensorFlow
             # uses hvd.DistributedOptimizer() to compute gradients.   
             model.compile(optimizer=opt,
                         loss = tf.losses.BinaryCrossentropy(),
                         metrics = [tf.metrics.BinaryAccuracy(name='accuracy')],
                         experimental_run_tf_function = False
                          )
             
             return model
         
         callbacks = []
             
         # Step 5: broadcast initial variable states from the first worker to 
         # all others by adding the broadcast global variables callback.
         callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
         
         # Step 7: average the metrics among workers at the end of every epoch
         # by adding the metric average callback.
         callbacks.append(hvd.callbacks.MetricAverageCallback())
         
         if args.use_checkpointing:
             # TensorFlow normal callbacks
             callbacks.apped(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min'))
             
             # Step 8: checkpointing should only be done on the root worker.
             if hvd.rank() == 0:
                 callbacks.apped(tf.keras.callbacks.TensorBoard(args.logdir/name_of_model))
         
         # Step 10: implement a LR warmup over `args.warmup_epochs`
         callbacks.append(hvd.callbacks.LearningRateWarmupCallback(initial_lr = args.base_lr, warmup_epochs=args.warmup_epochs, verbose=verbose))
             
         # Step 10: replace with the Horovod learning rate scheduler, 
         # taking care not to start until after warmup is complete
         callbacks.append(hvd.callbacks.LearningRateScheduleCallback(initial_lr = args.base_lr, start_epoch=args.warmup_epochs, multiplier=lr_schedule))
         
         
         # Creating model
         model = create_model(module_url, embed_size=embeding_size, name=name_of_model, trainable=True)
         
         start = time.time()
         
         if hvd.rank() == 0:
             print("\n##-------------------------##")
             print("Training starts ...")
         
         history = model.fit(train_dataset,
                             # Step 6: keep the total number of steps the same despite of an increased number of workers
                             steps_per_epoch = (train_df.shape[0]//args.batch_size ) // hvd.size(),
                             # steps_per_epoch = ( 5000 ) // hvd.size(),
                             workers=args.num_worker,
                             validation_data=valid_dataset,
                             #Step 6: set this value to be 3 * num_test_iterations / number_of_workers
                             validation_steps = 3 * (valid_df.shape[0]//args.batch_size ) // hvd.size(),
                             # validation_steps = ( 5000 ) // hvd.size(),
                             callbacks=callbacks,
                             epochs=args.epochs,
                             # use_multiprocessing = True,
                             verbose=verbose)
         
         endt = time.time()-start
         
         if hvd.rank() == 0:
             print("Elapsed Time: {} ms".format(1000*endt))
             print("##-------------------------##")

The following steps need to be performed before running this example:

.. important::
   :class: dropdown 

    **Prerequisites**

    - For Kebnekaise:
    
      ml GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5
      ml TensorFlow/2.4.1
      ml Horovod/0.21.1-TensorFlow-2.4.1

      virtualenv --system-site-packages /proj/nobackup/<your-project-storage>/env-horovod

      source /proj/nobackup/<your-project-storage>/env-horovod/bin/activate

      python -m pip install  tensorflow_hub

      python -m pip install  sklearn

A sample batch script for running this Horovod example is here:


.. code-block:: sh 

    #!/bin/bash
    #SBATCH -A project_ID
    #SBATCH -t 00:05:00
    #SBATCH -N X               # nr. nodes
    #SBATCH -n Y               # nr. MPI ranks
    #SBATCH -o output_%j.out   # output file
    #SBATCH -e error_%j.err    # error messages
    #SBATCH --gres=gpu:k80:2
    #SBATCH --exclusive
     
    ml purge > /dev/null 2>&1
    ml GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5
    ml TensorFlow/2.4.1
    ml Horovod/0.21.1-TensorFlow-2.4.1
      
    source /proj/nobackup/<your-project-storage>/env-horovod/bin/activate
       
    list_of_nodes=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:4,/g' )
    list_of_nodes=${list_of_nodes%?}
    mpirun -np $SLURM_NTASKS -H $list_of_nodes python Transfer_Learning_NLP_Horovod.py --epochs 10 --batch-size 64

.. challenge:: Running the Horovod example
    
    Do the initial steps for loading the required modules for Horovod, create 
    an environment and install the dependencies for Horovod. 

    Run the Horovod example on 1 node each with 4 GPU engines. Thus, 4 MPI ranks
    will be needed. Then run the script on 2 nodes. Compare the wall times reported
    at the end of the output files.
