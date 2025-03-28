## Spring 2023
### When?
- Tue 23 May 9-16

### Meeting 7 June 13:00
- Go through evaluation
    - Word document (by Björn) in Slack
- Learnings
    - Make most of morning session prereqs
        - Py-R-jl course
    - More time for type-along and exercises (no hurry)
    - Keep silent breakout room all the time



### Meeting Mon 22 May 10:00
- To discuss
    - status
    - registered: 84 (3 no account yet)
    - jupyter at UPPMAX rephrase
    - björn moves preparation/practicals
    - py-R-jl course in Oct
    - HPC-python in Dec
    - Birgitte tests ML extra exercise
    - Birgitte tarballs of exercises


### Meeting Wed 17 May 9:30
- To discuss
    - status


#### Changes of content (Björn's suggestions)


- Extra packages
    - [x] setup.py
    
- isolated (only Birgitte, because conda is in packages)
    - [x] Björn can change venv to virtualenv for UPPMAX?
    - [x] rm setup.py
- extra isolated
    - [x] link to course: make own packages


### Meeting Fri 12 May 13:15
- To discuss
    - status
    - topics
    - schedule
    - FAQS (or implement in material) from previous Q/A
    - gpu on UPPMAX test
        - GPU + ML exercises
    - fix julia interaction on UPPMAX
        - Björn Look
    - reservations
        - magnetic reservation GPU

#### Reservation
- tell in intro
- https://support.naiss.se/Ticket/Display.html?id=274512
- ``scontrol -M snowy show res naiss2023-22-500``

#### Topics
- "Syllabus"
    - add: practicals here from Intro
    - mv to extra:
    - comments: wait with schedule until 17 May
- Intro
    - add: paths to tarboll when avail
    - mv to extra: sites and hardware
    - comments:
- Load-run
    - add: (updates of versions)
    - mv to extra: nope
    - comments:
- packages 
    - add: pip install exercise
    - mv to extra: 
    - comments: keep because of goal of course
- isolated
    - add:
    - mv to extra: "self-installed..."
    - comments: venv also at HPC2N, keep (goal of course )
        - in extra: point to develop packages yourself links
- batch
    - add:
    - mv to extra:
    - comments: In order! (minor fixes?)
- interactive
    - add: jupyter speed up in thinlinc
    - mv to extra: Jupyter 
        - (UPPMAX vs HPC2N tabs
    - comments: 

- parallel
    - add: 
    - mv to extra: nope
    - comments: (minor fixes?)
- GPU
    - add: demo2 --> exercise
    - mv to extra: nope
    - comments: (checking)
- ML 
    - add:
    - mv to extra:
    - comments: merge ML 


#### Timings
|Session| Feb23 | Suggestions May23 | 
|-------|-------|----------|
|"Syllabus"| 15 | 10   |
|Intro| 10 | 10
Load-run| 15 | 15
packages| 20+15 conda | 35
isolated |  30 | 20
batch| 25 | 25
interactive| 15 | 20
**sum**| 145| 135 |
(add exercises)
Lunch??

|Session| Sep22 (in practice) | Suggestions May23 | 
|-------|-------|----------|
parallel| 36| 40|
GPU|17 | 30 | 
ML (incl all exercises)|21+30 | 40
Extra exercise time| |15
Sum| |5
Q/A ||20
**sum**| 104| 150



### Meeting Thu 4 May 11:00
- To discuss
    - follow up
    - issues

#### Follow up
- [x] Web pages
- [x] Invitation by Thu
- [x] Form by Thu
- [x] Make a branch for old course
    - Björn
- [x] Remove site specific 
    - Björn 
- [x] Course project
    - Birgitte
- [x] Issues
    - but otherwise work directly in main
- [x] Copy files from python feb23 course
    - Björn
- [x] integrated shared document questions into *each* sessions
    - [x] all teachers
    - [x] in text
    - [x] FAQ:s
- [x] Test the code on UPPMAX
    - (there are updates from julia-python wrt gpu)
    - ML
- [x] GPU
    - integrate from Feb
- [ ] Extra reading
    - [x] files (e.g. "packages_deeper")
    - [ ] content
- [x] Timings (when first versions of updates are done)
    - Björn

 
Q/A page from the R/Julia/Python course: https://umeauniversity.sharepoint.com/:w:/s/HPC2N630/EbHWglWYU_VNpTpdD2CtSfYBlpsAF6DyD_4RMwWCie_B0g?e=UddlVs

### Meeting Mo 24 Apr 10:30
- NAISS
    - course project UPPMAX
    - if already account on HPC2n let's keep
- reservations
    - snowy reservations for GPU (some hours)
    - rackham for cpu runs for a period of time
- Learnings from last time(s)
    - HPC python
        - check evaluation
    - Python-R-julia
        - use updates
        - check evaluation
- Teachers
- How?
   - Still seperate sessions
       - No!
       - **tabs instead**
   - record?
       - Yes!
- Material
    - still much packaging?
        - a little bit less
    - more time for hands-on
    - packaging and isolated
        - briefly go through
        - put deeper material from python-julia-R as "optional" but included in RTD material
            - Björn
    - very briefly on Bianca but point to special bianca course
    - ML on UPPMAX
      - Diana/Björn test?
    - intro
        - NAISS and the different centres overview
        - links to other NAISS python web pages
        - beginning? or for each section?
            - purpose
            - how to load
            - conda or not?

    - Exercise rooms
        - main
        - silent
        - hpc2n
        - several help rooms

    - Shared document
        - sharepoint 365 (Umeå hosts)

### Lecturer

|Session| Björn | Birgitte | Pedro|
|-------|-------|----------|------|
|"Syllabus"+intro|  | X| X
Load-run| X
packages| X
isolated | conda| pip
batch| | X
interactive| X
parallel| | | X
GPU| |  | X
ML (incl exercises)| | X
Exercises (exercises)| several different rooms |x|x
Question session (breakout)| X | X | X
Summary| X | X | X 

### Until next time
- [x] Web pages
- [x] Invitation by Thu
- [x] Form by Thu
- [ ] Timings
    - first see what is left after moving things to extra material
    - Björn
- [x] Make a branch for old course
    - Björn
- [ ] Copy files from python feb23 course
    - Björn 
- [x] Issues
    - but otherwise work directly in main
- [x] Course project
    - Birgitte
- [ ] Test the code on UPPMAX
    - (there are updates from julia-python wrt gpu)
    - ML
- [ ] GPU
    - integrate from Feb
    - 
### Invitation text
- [x] Thursday



**Online workshop: "Python in an HPC environment", *May 23*, 2023**

UPPMAX and HPC2n is organising a joint workshop on how to run Python and install additional Python packages on HPC resources and how to use the HPC capabilities. Participants will be able to bring their particular software request for discussion as well.

The instructors will use UPPMAX's systems for demos and there will be hands-on exercises for the participants.

**The following will be covered:**
* Loading the appropriate Python module 
* Packages
    * checking installed packages
    * pip install
    * Conda install
* Isolated environments
* HPC
    * batch jobs with Python
    * parallel computing
    * GPU computing
    * Machine Learning environment

Prerequisities:
- basic Linux 
- basic Python
- basic Slurm

When: Tue, May 23, 2023.
Time: 9:15 - 12:00, 13.15-16.00
Where: online.

Course web pages

### Outline
#### Common
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
##### LUNCH break
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
        - Machine Learning given by HPC2N
