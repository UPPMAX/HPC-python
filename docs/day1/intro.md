# Introduction #

<dl>
  <dt>`Welcome page and syllabus <https : //uppmax.github.io/HPC-python/index.html>`_</dt>
  <dd>- Also link at House symbol |:house:| at top of page </dd>
</dl>
<dl>
  <dt>   </dt>
  <dd>
    <p></p>
    <p>- Load Python modules and site-installed Python packages</p>
    <p>- Create a virtual environment</p>
    <p>- Install Python packages with pip (Kebnekaise, Rackham, Snowy, Cosmos)</p>
    <p>- Write a batch script for running Python</p>
    <p>- Use the compute nodes interactively</p>
    <p>- Use Python in parallel</p>
    <p>- Use Python for ML</p>
    <p>- Use GPUs with Python</p>
    <p></p>
  </dd>
</dl>
## What is python? ##

<dl>
  <dt>As you probably already know…</dt>
  <dd>
    <p></p>
    <p>- “Python combines remarkable power with very clear syntax.</p>
    <p>- It has modules, classes, exceptions, very high level dynamic data types, and dynamic typing.</p>
    <p>- There are interfaces to many system calls and libraries, as well as to various windowing systems. …“</p>
  </dd>
</dl>
In particular, what sets Python apart from other languages is its fantastic
open-source ecosystem for scientific computing and machine learning with
libraries like NumPy, SciPy, scikit-learn and Pytorch.

- `Official Python documentation <https://www.python.org/doc/>`_ 
- `Python forum <https://python-forum.io/>`_
- `A nice introduction to packages <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_
- The youtube video `Thinking about Concurrency <https://www.youtube.com/watch?v=Bv25Dwe84g0>`_ is a good introduction to writing concurrent programs in Python
- The book `High Performance Python <https://www.oreilly.com/library/view/high-performance-python/9781492055013/>`_ is a good resource for ways of speeding up Python code.
  ### Material for improving your programming skills ###
  

---

<dl>
  <dt>   `The Carpentries <https : //carpentries.org/>`_  teaches basic lab skills for research computing.</dt>
  <dd>
    <p>`The Carpentries <https://carpentries.org/>`_  teaches basic lab skills for research computing.</p>
    <p>- `Programming with Python <https://swcarpentry.github.io/python-novice-inflammation/>`_</p>
    <p>- `Plotting and Programming in Python <http://swcarpentry.github.io/python-novice-gapminder/>`_</p>
    <p>General introduction to Python by UPPMAX at https://www.uu.se/en/centre/uppmax/study/courses-and-workshops/introduction-to-uppmax</p>
  </dd>
</dl>
<dl>
  <dt>   Other course/workhops given by NAISS HPC centres : </dt>
  <dd>
    <p>Other course/workhops given by NAISS HPC centres:</p>
    <p>- `Pandas by LUNARC <https://github.com/rlpitts/Intro-to-Pandas>`_</p>
    <p>- `Matplotlib for publication <https://github.com/rlpitts/Matplotlib4Publication>`_</p>
    <p>CodeRefinery develops and maintains training material on software best practices for researchers that already write code. Their material addresses all academic disciplines and tries to be as programming language-independent as possible.</p>
    <p>- `Lessons <https://coderefinery.org/lessons/>`_</p>
    <p>- `Data visualization using Python <https://coderefinery.github.io/data-visualization-python/>`_</p>
    <p>- `Jupyter <https://coderefinery.github.io/jupyter/>`_</p>
    <p>Aalto Scientific Computing</p>
    <p>- `Data analysis workflows with R and Python <https://aaltoscicomp.github.io/data-analysis-workflows-course/>`_</p>
    <p>- `Python for Scientific Computing <https://aaltoscicomp.github.io/python-for-scicomp/>`_</p>
    <p>- `Introduction to packages <https://aaltoscicomp.github.io/python-for-scicomp/dependencies/>`_</p>
  </dd>
</dl>
<dl>
  <dt>   `ENCCS (EuroCC National Competence Centre Sweden) <https : //enccs.se/>`_ is a national centre that supports industry, public administration and academia accessing and using European supercomputers. They give higher-level training of programming and specific software.</dt>
  <dd>
    <p>`ENCCS (EuroCC National Competence Centre Sweden) <https://enccs.se/>`_ is a national centre that supports industry, public administration and academia accessing and using European supercomputers. They give higher-level training of programming and specific software.</p>
    <p>- `High Performance Data Analytics in Python <https://enccs.github.io/hpda-python/>`_</p>
    <p>- The youtube video `Thinking about Concurrency <https://www.youtube.com/watch?v=Bv25Dwe84g0>`_ is a good introduction to writing concurrent programs in Python</p>
    <p>- The book `High Performance Python <https://www.oreilly.com/library/view/high-performance-python/9781492055013/>`_ is a good resource for ways of speeding up Python code.</p>
    <p></p>
  </dd>
</dl>
## Documentations at other NAISS centres ##

<dl>
  <dt>.. seealso :  : </dt>
  <dd>
    <p>- LUNARC</p>
    <p>- `Python <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/>`_</p>
    <p>- `Jupyter <https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/?query=jupyter#jupyter-lab>`_</p>
    <p>- C3SE</p>
    <p>- `Python <https://www.c3se.chalmers.se/documentation/applications/python/>`_</p>
    <p>- `Jupyter <https://www.c3se.chalmers.se/documentation/applications/jupyter/>`_</p>
    <p>- `Python <https://www.nsc.liu.se/software/python/>`_</p>
    <p>- PDC</p>
    <p>- `Python <https://support.pdc.kth.se/doc/software-docs/python/>`_</p>
  </dd>
</dl>
<dl>
  <dt>.. important :  : </dt>
  <dd>
    <p>Project ID and storage directory</p>
    <p>- UPPMAX:</p>
    <p>- Project ID: naiss2024-22-1442</p>
    <p>- Storage directory: /proj/hpc-python-fall</p>
    <p>- HPC2N:</p>
    <p>- Project ID: hpc2n2024-142</p>
    <p>- Storage directory: /proj/nobackup/hpc-python-fall-hpc2n</p>
    <p>- LUNARC:</p>
    <p>- Project ID: lu2024-2-88</p>
    <p>- Storage directory: /lunarc/nobackup/projects/lu2024-17-44</p>
    <p>- NSC:</p>
    <p>- Project ID: naiss2024-22-1493</p>
    <p>- Storage directory: /proj/hpc-python-fall-nsc</p>
    <p>Login to the center you have an account at, go to the storage directory, and create a directory below it for you to work in. You can call this directory what you want, but your username is a good option. </p>
  </dd>
</dl>
<dl>
  <dt>.. important :  : </dt>
  <dd>
    <p>Course material</p>
    <p>- You can get the course material, including exercises, from the course repository on GitHub. You can either (on of these):</p>
    <p>- Clone it: ``git clone https://github.com/UPPMAX/HPC-python.git``</p>
    <p>- Download the zip file and unzip it:</p>
    <p>- ``wget https://github.com/UPPMAX/HPC-python/archive/refs/heads/main.zip``</p>
    <p>- ``unzip main.zip``</p>
    <p>- You should do either of the above from your space under the course directory on the HPC center of your choice. </p>
  </dd>
</dl>
<dl>
  <dt>    We will : </dt>
  <dd>
    <p>We will:</p>
    <p></p>
    <p>- teach you how to navigate the module system at HPC2N, UPPMAX, LUNARC, and NSC</p>
    <p>- show you how to find out which versions of Python and packages are installed</p>
    <p>- look at the package handler **pip**</p>
    <p>- explain how to create and use virtual environments</p>
    <p>- show you how to run batch jobs</p>
    <p>- show some examples with parallel computing and using GPUs</p>
    <p>- guide you in how to start Python tools for Machine Learning</p>
  </dd>
</dl>
 
