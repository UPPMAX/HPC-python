# 18 May
- Content
- line-up
- Letter info
- Workshop page

## How?
- one full day
- workshop
- Fri 2 YES
- Tue 6 sep? NO
- (7 Sep maintenance)

## Learning outcomes
- load and run python
- find installed packages
- install package with pip
- install package with conda
- create a venv
- use a venv
- jupyter on calculation node (more or less about)
    - HPC2N perhaps working but different approach? Pedro looks up
- run batch jobs with python code
- run parallel
    - calc pi with mpi
- run workflows
    - how much??
- load and use ML packages
    - tensor flow example (UPPMAX)
- gpu
## Learning experience
- webinar
- pair-teaching
- code along
- problem-based approach
    - find out an example or two?
- group or pair work? TBD
    - exercise
- more?
## Prerequisites
- basic Bash commandline + scirpting
    - explicit
- basic Python
    - 
- basic Slurm

## Content
### Morning
- Python modules (LMOD)
- python packages
- installations
- Kebnekaise/Bianca separated session 	
    - user specific session as well
- workflow
- 
### Afternoon
- interactive 
- jupyter on calculation nodes
- batch
- parallel computing
- tensorflow??
- Problem-based approach
    - how to approach a problem at the site to solve it
    - a number of example problems
        - OOM-related (out of memory?)
    - numerical- as well as data/ I/O-based
- check how the resources are used
    - graphana
    - jobstats

## Plan
| Time | Topic | Activity | Description | Goal |
| -----| ----- | -------- |------------ | ---- |
**Morning 2 hours**| 45 + 40m
10	|Intro	|Lecture |	Outline	| Get overview and LOs
10	|Load and run|	Lecture+code along	| Show modules |	Learn how to load a python version
25	|packages	|Lecture + code along + (exercise)	|Check current, Pip, Conda | 	List packages, do pip installation,do conda installation 
10	|Isolated environ	|Lecture + code along |	Venv |	Understand virtual environ
25	| Kebnekaise/Bianca separated session| Lecture | Cluster specific practice and installed packages, User interaction | Understand cluster limitations		
5	|Summary|	Lecture|	Describe when to do what|	Keypoints
**Afternoon 2 hours** | 45+45m
30|   batch with python and conda |Lecture+code along + exercise | python with batch | write batch script with right envs
15|   interactive jupyter| Lecture + code along +exercise | run jupyter on calculation nodes| run jupyter on calculation nodes
30|   workflows?
15|  Summary


## Letter info
- [ ] Sep 2
- [ ] Sep 6 
- [X] Sep 9

**Online workshop: "Python at UPPMAX & HPC2N", *Sep 9*, 2022**


UPPMAX and HPC2n is organising a joint workshop on how to run Python and install additional Python packages on the computer resources provided by UPPMAX and HPC2N and how to use the HPC capabilities. Participants will be able to bring their particular software request for discussion as well.

**The following will be covered:**
* Loading the appropriate Python module 
* Packages
    * checking installed packages
    * pip install
    * Conda install
* Virtual environments
* Kebnekaise / Bianca session
* HPC
    * interactive jupyter
    * batch jobs with Python
    * Machine Learning environment
    * workflows

Prerequisities:
- basic Linux 
- basic Python
- basic Slurm



When: Friday, Sep 9, 2022.
Time: 9:15 - 12:00, 13.15-16.00
Where: online.

Registration opens in June

### SNIC training letter
- 25 May (Wed)


## Web page later




# 21 April

## To be decided today
- [x] Content
- [x] Just UPPMAX/HPC2N?
- [x] Length
- [x] Style (one-way, workshop, course)
- [x] One or two instances
- [x] Month (at least)
- [x] Who wants to contribute

## Intro
- Pedro asked:
"I thought we could join efforts and collaborate in a Python course where we could talk about the workflows at UPPMAX and HPC2N? 
If you think this is doable, maybe we could run it in the summer/autumn? "


## Content
- core from “Python at Uppmax” and made general
      - https://python-at-uppmax.readthedocs.io/en/main/
      - https://uppmax.uu.se/support/user-guides/python-user-guide
  HPC

:::warning
same at HPC2N? (except Bianca)
- not Conda
:::

- More workflows
    - snakemake?
        - can submit slurm jobs
    - Own workshop?

- Hpc possibilities 
    - Focus on site-specific systems
        - modules
        - sbatch
        - interactive
    - parallel computing
    - GPU?
    - TensorFlow
- More…?
    - jupyter on calculation nodes??
- Problem-based approach
    - how to approach a problem at the site to solve it
    - a number of example problems
        - OOM-related
    - numerical- as well as data/ I/O-based
    - check how the resources are used
## What other courses/workshops/training are there? Not to collide (*in content*) with:
- Python intro course at UPPMAX
    - no mpi4py there
- ENCCS: **HPDA Python**, https://enccs.se/events/2022-05-hpda-python
    - **May** (closed)
    - tools for performant processing 
        - (netcdf, numpy, pandas, scipy) on single workstations 
    - **parallel**, **distributed** and **GPU** computing 
        - (**snakemake**, numba ( just-in-time compiler for Python), **dask**, **multiprocessing**, **mpi4py**).
- CodeRefinery: **Python for Scientific Computing** course together with Aalto University and CodeRefinery
    - https://aaltoscicomp.github.io/python-for-scicomp/
    - 4 half-days
    - dates not yet set, most likely in **October**
    - **mpi4py**
    - **distrib**
    - **packages**
- CodeRefinery workshop 
    - **Sep 20-22 and 27-29**,  (6 half-days)
    - online
    - Git, collab version control
    - reproducible research, social coding
    - jupyter
    - **workflows with snakemake** , and more
    - software testing (**pytest**)
    - modular code devel

- PDC Summer School 2022: Introduction to High Performance Computing, **15-26 August 2022**
    - https://www.pdc.kth.se/summer-school/2022
    - **No python??**
    - parallel algorithms 
    - parallel programming 
    - modern HPC architectures 
    - performance analysis and engineering 
    - software engineering 
- Earlier SNIC parallel courses (**Dec 2020**)
    - SNIC Intro to MPI (C, Fortran, **Python**)
        - https://github.com/SNIC-MPI-course/MPI-course
- PDC/ENCCS: https://pdc-support.github.io/introduction-to-mpi/index.html
    - don't know the status
- more?
    - NBIS snakemake (general)
    - https://nbis-reproducible-research.readthedocs.io/en/course_1911/snakemake/
    - 
## Prerequisites
- basic Slurm
- basic Bash scripting
- basic Python

## How?
- [x] Divided into 
    - [x] packages and 
    - [x] HPC python computing
- [x] One instance
- [ ] Seminar(s)
- [x] Workshop
- [ ] Hackaton? Another time?
- [ ] Course?
- [x] Also updated documentation
- [X] new topics/examples gradually 
## Length?
- [ ] 3 hours
- [x] Full day
- [ ] Several days?

## When?
- [ ] Python-at-UPPMAX+HPC2 May 18 10-12
- [ ] Aug, probably not
- [x] Sep 2, Sep 6-8 somewhere
- [ ] Oct
## Who?
- Any other SNIC centre?
    - not this time
    - they can learn from
## Other 
- 
## Next steps
- Further planning
    - Content
    - line-up
    - Letter info
    - Workshop page
- Who wants to be part?
    - Björn
    - (Lars)
    - Pedro
- SNIC training letter
    - 28 April
    - 12 May
    - 25 May (Wed)
## Next meeting
- 19 or 20 May, Doodle
- 

```python=
import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD 
size = comm.Get_size()
rank = comm.Get_rank()
```
