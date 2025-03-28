# Meeting notes autumn 2022 part 2

## Meeting 7 Sep
- status
- ideas
- todo:s
- the recording
    - is editing afterwards necessary?
    - primary speaker view
    - settings in meeting before
    - mute at entrance

### Status?
- Björn keeps filling in UPPMAX info and works with formatting
- in addition we need better merges in some sessions
    - load-run
    - isolated general
    - interactive



### Ideas
- many small breakoutrooms
    - many hosts (helpers)
    - also ask for help in HackMD
- install numba, mpi4py etc in Isolated envs sessions? 
    - Maybe not, but virtual env name created in Isolated sessions and with name connected to course name
- UPPMAX users will be able to run CPU versions of ML in exercises unless they already have
- I have booked Snowy for a demo using the GPUs in the general GPOU session

### Todo:s
[x] Björn will make HackMD for course

- Introduction
    - HPC2N fill in table: Mem/node+GPU

- evaluation form
    -  Birgitte



## More
- inform about needs from helpers
    - HackMD
    - chat
    - breakoutrooms
        - during exercise
        - technical problems
- identify helpers

## Background

98 registered

### Experience with LINUX 

Orientation	94%
Bash	    86%
Batch	    75%
Packages	87%

### Cluster (multiple)

HPC2N 47/84	    55
Rackham 47/84	56
Bianca 18/84	21
Snowy 14/84		17

### Size of your research problem (multiple)
Single 25 %
**Node 63%**
Multinode 41%
Don’t know 25%

## Suggested outline
### Common
- Syllabus
- Intro (incl. very briefly about UPPMAX/HPC2N)
    - terminal vs thinlinc (up to users)
- Load/Run
- Packages especially pip
- Isolated general
- Interactive general 
    - rm `-p`
- Batch general
    - this one before interactive
- Parallel session 1
    - UPPMAX: Conda, venv (exercises), 
    - Kebnekaise: virtual env (exercises)
#### LUNCH break
- Parallel computing general
    - also exercises
- GPU general
    - only demos
        - snowy
- ML general
    - some links to UPPMAX
    - cpu exercise
- Parallel session 2
    - UPPMAX: Bianca, jupyter
    - HPC2N: 
        - gpu exercises at Kebne
        - **Machine Learning given by HPC2N**
        
   
- Summary

## Time estimations (min)

| Time (UPPMAX) | Time (HPC2N) | Topic | Activity | Description | Goal |
| -----| ----- | -------- |------------ | ---- |--|
**Morning 3 hours**| 50 + *45m* +*40*
10  |9.00  |Syllabus
15	|  |Intro	|Lecture |	Outline	| Get overview Cluster+Python and LOs
15	|  |Load and run|	Lecture+code along	| Show modules |	Learn how to load a python version
15	|  |packages intro	|Lecture *+ code along + (exercise)*	|Check current, Pip | 	List packages, do pip installation
|**15**| 9.55|**Coffee**||
15	| 10.10 |Isolated Intro	|Lecture + code along |	 |	Understand virtual environ
20	|  |batch|
15	|  |Interactive  |
5| 11.00 |**short leg stretch**
45| 11.05|UPPMAX session1| Lecture +Ex? |Conda, isolated, 
45| |HPC2N session1|Lecture + Ex?|Kebnekaise,isolated|  Cluster specific practice and installed packages, User interaction |
|**Afternoon 3 hours** | *45+45m*| |
30	|13.00  |parallel| +exercises
20   |  |gpu| +demos
5|13.50 | **Leg stretch**
20	| 13.55 |ML intro|+demos
25| 14.15 |UPPMAX session2| Lecture +Ex? | jupyter, Bianca
25| 14.15|HPC2N session2|Lecture + Ex?|ML exercise|  User interaction | 		
|**15**|14.40 |**Coffee**||
15| 14.55	|Summary|	Lecture|Describe when to do what|	Keypoints
25| 15.10| Exercises not recorded
25|15.35|Questions not recored

## Lecturer


|Session| Björn | Birgitte | Pedro|
|-------|-------|----------|------|
|"Syllabus"| X | X| (X)
|Intro incl cluster| X | X | (X)
Load-run| X
packages| X
isolated intro| X
batch| | X
interactive| X
parallel| | | X
GPU| |  | X
ML| | X
UPPMAX1 (Co,Is,)| X
HPC2N1 (Keb,Is )| | X|
UPPMAX2 (Bi, (Jupyter)| X
HPC2N2 (ML exercises)| | X| X
Exercises (exercises)| exercise rooms and help rooms
Question session (breakout)| X | X | X
Summary| X input | X | X input

## To discuss

- parallel
    - copy paste exercises or given files?
- gpu 
    - only HPC2N?
    - demo on Snowy


## Course material

### General
- shorter titles in drop-down 
  - [Link](http://www.thing.com) (http://www.thing.com)
  - move split sessions to afternoon

- index.rst
    - [name=Björn]
- Intro
    - [name=Björn]
    - only overview
    - move "facts" to sections
    - merge more and split if different (true for all section)
- load_run.rst
    - [name=Björn]
    - Load
        - tabs
    - Run
        - tabs
        - be clear what is meant by interactive
- packages.rst
    - [name=Björn]
    - tabs
    - install with conda (move to UPPMAX)
- isolated.rst
  - [name=Björn]
      - split and insert to site-specific session
      - pyenv only for uppmax
- bianca.rst
  - [name=Björn] 
  - format
  - demo pip wharf
- kebnekaise
  - [name=Birgitte]
  - [name=Pedro]
- interactive 
  - [name=Björn] 
  - [name=Birgitte]
  - [name=Pedro] 
  - Introduce batch already here?
- batch 
  - [name=Björn]
  - [name=Birgitte]
  - [name=Pedro] 
- parallel computing
  - [name=Pedro] 
  - [name=Björn]
- ml
  - [name=Birgitte]
  - [name=Björn]
  - split to sessions ()
- gpu
  - [name=Birgitte]
  - [name=Pedro] 
  - [name=Björn]
- summary.rst
  - [name=Björn]
  - [name=Birgitte]
  - [name=Pedro] 

## Latest version Letter

**Using Python in an HPC environment, UPPMAX/HPC2N September 9 2022. Online workshop**

UPPMAX and HPC2N are organising a joint workshop on how to run Python codes and install additional Python packages on the computer resources provided by these two HPC centres. Participants are encouraged to bring their particular software request for discussion as well.

The goal for the course is that you will be able to:

- Load Python modules and site-installed Python packages
- Create a virtual environment
- Install Python packages with pip (Kebnekaise, Rackham, Snowy)
- Install Python packages with conda (Bianca)
- Write batch scripts for running Python
- Use Python in parallel
- Use Python for ML
- Use GPUs with Python

Prerequisites: familiarity with the LINUX command line, basic Python, and batch jobs. 

For more info and registration, please visit https://docs.uppmax.uu.se/courses_workshops/hpc_python/ or https://www.hpc2n.umu.se/events/courses/2022/python-in-hpc

