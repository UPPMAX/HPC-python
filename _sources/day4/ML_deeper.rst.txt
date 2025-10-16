More about ML
=============

This section contains one more example for ML. 

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

    - For Rackham/Snowy:

      module load python_ML_packages python/3.9.5 gcc/10.3.0 build-tools cmake/3.22.2

      cd /proj/hpc-python/<mydir-name>
      python -m venv --system-site-packages env-horovod

      source /proj/hpc-python/<mydir-name>/env-horovod/bin/activate

      pip install --no-cache-dir --no-build-isolation horovod

      pip install --no-cache-dir --no-build-isolation tensorflow-hub

A sample batch script for running this Horovod example is here:

.. tabs::

   .. tab:: UPPMAX

      .. code-block:: sh 

          #!/bin/bash -l
          #SBATCH -A naiss2024-22-415
          #SBATCH -t 00:05:00
          #SBATCH -M snowy
          #SBATCH -n 1
          #SBATCH -o output_%j.out   # output file
          #SBATCH -e error_%j.err    # error messages
          #SBATCH --gres=gpu:1

          # Set a path where the example programs are installed.
          # Change the below to your own path to where you placed the example programs
          MYPATH=/proj/hpc-python/<mydir-name>/HPC-python/Exercises/examples/programs/

          ml purge
          module load uppmax
          module load python_ML_packages python/3.9.5
          module load gcc/10.3.0 build-tools cmake/3.22.2

          # Change the below to your own path to the virtual environment you installed horovod to
          source /proj/hpc-python/<mydir-name>/env-horovod/bin/activate

          srun python $MYPATH/Transfer_Learning_NLP_Horovod.py --epochs 10 --batch-size 64


   .. tab:: HPC2N

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
