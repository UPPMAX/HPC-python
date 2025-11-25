# Using Python in an HPC environment

<!-- markdownlint-disable MD013 --><!-- Badges cannot be split up over lines, hence will break 80 characters per line -->

[![Check links](https://github.com/UPPMAX/HPC-python/actions/workflows/check_links.yaml/badge.svg?branch=main)](https://github.com/UPPMAX/HPC-python/actions/workflows/check_links.yaml)
[![Check spelling](https://github.com/UPPMAX/HPC-python/actions/workflows/check_spelling.yaml/badge.svg?branch=main)](https://github.com/UPPMAX/HPC-python/actions/workflows/check_spelling.yaml)

<!-- markdownlint-enable MD013 -->

This course aims to give a brief, but comprehensive introduction to using Python in an HPC environment. You will learn how to use modules to load Python, how to find site installed Python packages, as well as how to install packages yourself. In addition, you will learn how to use virtual environments, write a batch script for running Python, use Python in parallel, and how to use Python for ML and on GPU:s. 

The course is a cooperation between UPPMAX (Rackham, Snowy, Bianca), LUNARC (Cosmos), and HPC2N (Kebnekaise) and will focus on the compute systems at all 3 centres. For the site-specific part of the course you will be divided into groups depending on which center you will be running your code, as the approach is somewhat different. 

# Build locally  

Navigate to the root directory of your Sphinx project (where the conf.py file is located, ie. in `docs`) and run:

```cmd
make html
```

You can preview the generated HTML files locally by opening the _build/html/index.html file in your browser:

```cmd
xdg-open _build/html/index.html  # On Linux
open _build/html/index.html      # On macOS
```

To perform auto rebuilds and see changes on locahost:

```cmd
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
```


# Rendered website

- [Rendered website](https://uppmax.github.io/HPC-python/)

# Others

- [Meeting notes](meeting_notes/README.md)

