.. _common-naiss-projects-overview:

NAISS projects overview
=======================

These are the NAISS course projects:

+------------+--------------------------+
| HPC cluster| Course project           |
+============+==========================+
| COSMOS     | :code:`lu2025-7-106`     |
+------------+--------------------------+
| Dardel     | :code:`naiss2025-22-934` |
+------------+--------------------------+
| Kebnekaise | :code:`hpc2n2025-151`    |
+------------+--------------------------+
| Pelle      | :code:`uppmax2025-2-393` |
+------------+--------------------------+
| Tetralith  | :code:`naiss2025-22-934` |
+------------+--------------------------+
| Alvis      | :code:`naiss2025-22-934` | 
+------------+--------------------------+

Storage spaces for this workshop:

+------------+------------------------------------------------------------+
| HPC cluster| Course project                                             |
+============+============================================================+
| Alvis      | :code:`/mimer/NOBACKUP/groups/courses-fall-2025`           |
+------------+------------------------------------------------------------+
| Bianca     | None. Use ``/proj/[your_project_code]``                    |
+------------+------------------------------------------------------------+
| COSMOS     | :code:`/lunarc/nobackup/projects/lu2025-17-52`             |
+------------+------------------------------------------------------------+
| Dardel     | :code:`/cfs/klemming/projects/supr/courses-fall-2025`      |
+------------+------------------------------------------------------------+
| Kebnekaise | :code:`/proj/nobackup/fall-courses`                        |
+------------+------------------------------------------------------------+
| LUMI       | None. Use ``/project/[your_project_code]``                 |
+------------+------------------------------------------------------------+
| Pelle      | :code:`/proj/hpc-python-uppmax`                            |
+------------+------------------------------------------------------------+
| Tetralith  | :code:`/proj/courses-fall-2025/users/`                     |
+------------+------------------------------------------------------------+

.. admonition:: Reservations

   Include in slurm scripts with ``#SBATCH --reservation==<reservation-name>`` at most centers. (On UPPMAX it is "magnetic" and so follows the project ID without you having to add the reservation name.)

   **NOTE:** as there is only one/a few nodes reserved, you should NOT use the reservations for long jobs as this will block their use for everyone else. Using them for short test jobs is what they are for. 

   - UPPMAX 
       -   

   - HPC2N
       - ``hpc-python-fri`` for one AMD Zen4 cpu on Friday
       - ``hpc-python-mon`` for one AMD Zen4 cpu on Monday
       - ``hpc-python-tue`` for two L40s gpus on Tuesday
       - it is magnetic, so will be used automatically 

   - LUNARC 
       - ``hpc-python-dayN`` for up to 2 CPU nodes per day, where N=1 for Thursday, 2 for Friday, 3 for Monday, and 4 for Tuesday
       - ``hpc-python-day4-gpu`` for the GPU and ML sessions on Tuesday afternoon
       - **Note**: for On-Demand apps, click the gear icon next to "Resource" in GfxLauncher popup to see additional options, which should include a box to include a reservation. Ticking that box will reveal a dropdown menu with the list of reservations associated with the project.
