# Autumn

- 2-days

### Dates

- On-boarding Tue Dec 3, 1-2pm
- Thur-Fri Dec 5-6, 9-16

### Content

- 2nd day: Pandas more parallel/big data

### Meeting 11 Dec 14.15

- Agenda:
    - we have to draw up plans for NAISS training in 2025
    - please work on your plans for 2025.
    - dates should be given.
    - Joachim was instructed that plans for early in 2025 should be pretty precise. If the planing for the end of 2025 is more vague this would be understandable.

### Meeting 27 Nov at 14

#### Status
- Registrations 59
    - ``OS
Windows: 26
Linux: 17
macOS: 16``
    - ``Python knowledge:
None: 7
Basic/beginner: 15
Intermediate: 27
Proficient/expert: 10``
- Projects
    - storage on its way for LUNARC
- Continuous Integration
    - spelling in order
    - links needs revisit
        - manually fix for missing ``-matlab`` in 4-day course links
    - rst linter 
        - not activated yet
- Working code on all centres
    - Björn/Jayant has not tested yet
    - Birgitte half-ways
    - don't forget Tetralith
- Prerequisites
    - 2 files
    - [10  min course](https://pandas.pydata.org/docs/user_guide/10min.html)
    - "Preview" [cheat-sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- lessons
    - syllabus/index 20
        - keep log in and exercise fix here
    - load+run+scripts+import 30
        - works everywhere
    - packages/venvs (reducing time) 50--> 30
        - just needed thinks
        - exercise including making the env
        - please help to define packages needed on different systems (issue)
    - compute nodes 
        - show how to load  jupyter, VScode (Pedro from local), spyder (Rebecca)
        - on-demand (10-15 min)
        - split
            - batch 30
            - interactive (seq) 20
            - IDE 15 
            - users starts interactive for the afternoon 5h
    - matplotlib/analysis
        - spyder
    - pandas 
        - add for summary [cheat-sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) 

#### Discuss schedule
- Too much material?
- Change timings?

#### ToDos
- Content
- Test code
    - all centres
- tabs for
    - lunarc
    - nsc

### Meeting Nov 20

#### Status
- Registrations
    - 40
        - ~5 at lunarc
        - 8-10 at UPPMAX
        - ~5 Umeå/Luleå
        - ~20 at NSC
- Projects
    - 4 places
    - nsc for new users
- Content
    - added NSC to many parts
    - jupyter
    - we are responsible for our own sessions that is works everywhere
    - IDES presented before lunch
    - Rebecca focuses on using the tools
        - jupyter format--> md/rst
    - VSCode is on tetralith
        - just show how to load
        - let them choose if used later in course

    - GPU no much changes needed!
    - packages.rst may be updated for tetralith etc
        - more packages need to be installed by user
    - analysis day 2
        - conversion needed (perhaps link from Birgitte)
        - numba moved to parallelism
    - parallelism
        - identify needed packages
    - Big data: start work
    - DL: started
#### Discuss
- requirements for venvs 
    - different for different clusters
    - YES
- JUST Tetralith py/3.10.4 for all sessions
    - yes
- Exercise time: At least 25 %
    Votet YES

#### ToDos
- storage at Cosmos RP
- mermaid for NSC BC
- all define required packages for the python version we will use
- BC will collect
- let user define problem in text before course starts
- schedule fixed by wednesday
    - timings
    - no moving afterwards

next meeting: 14.00 Wednesday 27

### Meeting 13 Nov

#### Status
- Registrations
- ML
    - cleaned!
    - to start from
    - horovod in extra?


#### Discuss
- ML
    - do not show Theano (old)
    - keras together with TF (dependency)
    - needs updates (focus on 3.11.5/8 at UU)
        - pytorch and TF on Lunarc/HPC2N
        - sklearn at HPC2N
        - seaborn on LUNARC
- VSCode
    - test the limits and possibilities
- JuliaCall works in another way at tetralith

Next meeting
- Wednesday Nov. 20 at 14:00

### Meeting 8 Nov
#### Agenda

- Status
- Possible updates/plans

#### Status
- UPPMAX maintenance day before (Wed)
    - Involve Tetralith/Dardel? (must evidently do (this)
        - waiting for input from UPPMAX sysexps 
        - course project/staff
        - apply for tetralith! (GPU is available for us, it seems)
            - DONE!
            - dardel wait to include this time
- Registration
    - 32 registered
- Issues
    - prereqs: BC & RP rather well in advance
        - next week...
    - ML:
        - BB starts to look the status of the present stuff (by Tue)
        - then meet with Jayant
- Materials
    - Python >=3.11.5
    - Day 1 morning
    - Day 1 afternoon
        - Birgitte takes the GPU (Pedro away)
    - Day 2 morning
    - Day 2 afternoon

#### Possible updates/plans
- BC: Spyder can be installed in venv
    - need to check more details
    - EB only up to foss2020
    - anaconda is providers first choice but not my favorite!
    - on cosmos: on demand, type depencency in "additional text box"
    - RP check vscode how it is run? -Currently only on FE
    - Jayant runs vscode


#### ToDos
- look up bioconda
    - MIT license(?) it seems

### Meeting Oct 31
#### Agenda

- Status
- Evaluation of last sessions
- Schedules and teachers
- Possible updates/plans

#### Status
- Registrations are ongoing
    - 21
    - bio/atmos/astro/etc
- Advertisement
  - The goal for the course is that you will be able to
    - Load Python modules and site-installed Python packages
    - Create a virtual environment and install your own Python packages to it
    - Write a batch script for running Python
    - Use Python in parallel
    - Use Python for ML
    - Use GPUs with Python
    - Use pandas
    - Learn about matplotlib
  - Prerequisites: familiarity with the LINUX command line, basic Python
    - add pandas basics



#### Evaluations
**May course HPC-Python**
- Load and run		
    - Include in On-boarding? 
    - On Load/run session: Short summary. 
    - No recording at THAT time, interaction for those with problems
    - NOTES: 
        - On-boarding
            - PRESENT LOGIN AND MODULE LOADING
        - No login session 5-6 Dec?
        - or 9.00-10.00: login/load/editor/run/tar balls
        
- Install package at 10
    - Fine!?
- Batch mode: 		
    - Perfect! 
- Interactive on compute node (be clear en session title)
    - Let students try Jupyter in exercise
    - more hands-on
    - NOTE: Look into Spyder
- Parallel computing		
    - More hands-on
- GPUs			
    - Same material but faster? 
    - Or Extra material?
    - exercises
- ML			
    - More exercises?
    - NOTE: will extend!
    
More links to deeper material!!

**OBS! Above is just the old material**


#### Summary of earlier discussions
- 1st day: almost like before, shorten basics things
- 2nd day: Pandas more parallel/big data

##### Discuss the division of the days
- First day things not requiring batch and ML and GPU
    - ask compute node for interactive work
        - basic slurm
        - also GPU
    - Jupyter/spyder
    - pandas/matplotlib/seaborn?
- Second day: ML and parallelisms
    - 
- Vote YES
- Vote NO: BB
    - it is in the info already

### Preliminary schedule

Instead
- First day
    - 9.00 Login/load/run/tarball [name=Birgitte] + all 
    - 10.15 packages/virt envs (short) [name=Björn]
    - 11.15 basic slurm [name=Birgitte]
        - interactive 
        - get gpus
        - start jupyter/spyder
    - 13.00 analysis (75min) [name=Rebecca]
        - using IDE work environment 
            - jupyter/spyder(?)/VScode(?)
        - matplotlib 60
    - 14.30 GPU 60 [name=Pedro?]
    - 15.30 Use cases + Q/A [name=All]
    - 16.35 Ending with evaluation
- Second day 9-17
    - 9.00 Analysis 105m [name=Rebecca]
        - pandas 75
        - seaborn 30
    - 11.00 Parallelism 60-75 60+15m after lunch [name=Pedro]
        - MPI
        - dask
        - processes
    - 13.15 Big data 45 [name=Björn]
        - csv?
        - xarray?
        - netcdf
        - hdf5
        - chunking (dask+pandas?)
    - 14.15 ML+DL 2x45 min [name=Jayant]
        - pytorch
        - tensorflow
        - sklearn
    
    - 16.00 Use cases + Q/A [name=All]
    - 16.45 Ending with evaluation


- Evaluation in May?? (*already discussed above*)
    - big data
    - point to other on-line material for specific science topics
- To discuss further
    - jupyter
    - extend parallelism and ML
    - in application: **ask knowledge of parallel, slurm, gpu, ML**
    - more course links: https://enccs.github.io/gpu-programming/
    - rewrite course goal:
        - Deploy hpc-resources for different problems
            - not learning the solutions of the problems!

- Evaluation of Python day in Oct
- 

#### Lecturer

**Day 1**
|Session| Björn | Birgitte | Pedro|Rebecca
|-------|-------|----------|------|------|
|"Syllabus"+intro| X | | 
Load-run+packages| X
self-inst. isolated | | X | 
batch| | X |
parallel| | | X
interactive| X
GPU| |  | X
ML (incl exercises)| | X
Exercises (exercises)| several different rooms |x|x
Question session (breakout)| X | X | X
Summary| X | X | X 

**Day2**
|Session| Björn | Birgitte | Pedro|
|-------|-------|----------|------|


#### Old schedule **FIX**

|Session| May 2024 | Suggestions Dec | 
|-------|-------|----------|
|9.00 "Syllabus"| 30 | 15   |
|9.15 Intro| 10 | 10 |
9.25 Load-run + system inst. packages| 35 | 30 
9.55 **break**|15 |15
10.10 self-install. packages + isolated| 46 | 35
10:45 batch + arrays | 35 | 35
11.05 break during | 10 | 10
11.30 interactive + jupyter (together)| 15 (a little too short) | 25 |
12:00 Lunch

|Session| Dec23 | Suggestions 15-May-2024| 
|-------|-------|----------|
13.00 parallel| 55| 60 |
14:00 break | | 10
14:10 GPU | 30 | 40 | 
14.50 BREAK
15.05 ML (incl all exercises) | 23 | 30
15:35 Sum+eval| 18| 20
15:55 Q/A + extra exercise| 0 | 35
16:00 END-OF-DAY
**sum**| 150| 

#### ToDos
- look into spyder
    - installed on UPPMAX?
    - tunnelling YES
        - allowed at HPC2N? NO
        - 
    - 
