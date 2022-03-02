Load and run Python
===================

At UPPMAX we call the applications available via the module system modules. 
https://www.uppmax.uu.se/resources/software/module-system/ 

Load
----------
Load latest Python module by
module load python
Check all available version with:
module available python
Load specific version with:
module load python/X.Y.Z
Warning: Don’t use system-installed Python/2.7.5
ALWAYS use Python module

Note that you can run two python modules at the same time if one of the module names is python3/3.Y.Z and the other is python/X2.Y2.Z2.
Otherwise, the previous one will be unloaded

Sometimes necessary in pipelines and other toolchains where the different tools require different python versions.

Run
---

You can run a python script in the shell by:
python example.py
or, if you loaded a python3 module:
python3 example.py
You start a python session/prompt ( >>> ) by typing:
python or python3
ipython or ipython3 (interactive)
Exit with <Ctrl-D>, "quit()" or 'exit()’ in python prompt
