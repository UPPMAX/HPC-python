# Jupyter on compute nodes using browser at other place

```{callout} What is Jupyter?
  - "The Jupyter Notebook is the original web application for creating and sharing computational documents. It offers a simple, streamlined, document-centric experience."
    - Run python interactively and make a "story" document with text and code and figures woven together. 
    - Includes file manager
  - You run it in a **web browser** (``firefox`` at UPPMAX)
  - This may be slow unless you run the browser in ThinLinc or locally on you own computer.

  - The Jupyter project site contains a lot of information and inspiration. <https://jupyter.org/>
  - The Jupyter Notebook documentation. <https://jupyter-notebook.readthedocs.io/en/stable/>
  
```

## 1. Start an interactive session from the login node
``` console
# For Rackham
$ interactive -A <naiss-project> -n <cores>-t 4:00:00

# For Snowy
$ interactive -M snowy -A <naiss-project> -n <cores> -t 4:00:00
```

## 2. Start jupyter notebook (from python/3.10 also jupyter-lab) from the interactive session (*when it gets allocated*)
``` console
$ module load python/3.9.5
$ jupyter-notebook --ip 0.0.0.0 --no-browser
```

## 3. Connect to the running notebook 
Keep this session open and running. Note the node on which you are got the interactive job i.e. something like "r141".

### In ThinLinc

- If you are connected to Rackham via [ThinLinc session](http://docs.uppmax.uu.se/cluster_guides/thinlinc/), open a browser with the link you got but modified to point to r486 i.e. 
``http://r486:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``

### On own computer

- If you use ssh to connect to Rackham, you need to forward the port of the interactive node to your local computer.
    - On Linux or Mac this is done by running in another terminal. Make sure you have the ports changed if they are not at the default ``8888``.
        ``` console
        $ ssh -L 8888:r486:8888 <username>@rackham.uppmax.uu.se
        ```
    - If you use Windows it may be better to do this in the PowerShell instead of a WSL2.
    - If you use PuTTY - you need to change the settings in "Tunnels" accordingly (could be done for the current connection as well).
    ![](./img/putty.png)
    
    [SSH port forwarding](https://uplogix.com/docs/local-manager-user-guide/advanced-features/ssh-port-forwarding)
    

    On your computer open  the address you got but replace r486 with localhost i.e. 
``http://localhost:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``
or 
``http://127.0.0.0:8888/?token=5c3aeee9fbfc75f7a11c4a64b2b5b7ec49622231388241c2``

    This should bring the jupyter interface on your computer and all calculations and files will be on Rackham.
