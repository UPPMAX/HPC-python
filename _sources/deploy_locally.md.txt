# Deploy this repository as a local website

This page describes how to deploy this website locally.

To work on the website locally first create a virtual environment and install
the required dependencies:

???- question "Do I really need a virtual environment?"

    No.

``` bash
python -m venv hpc_python_venv
source hpc_python_venv/bin/activate
pip install -r requirements.txt
```

Then serve the website and edit

``` bash
mkdocs serve
```
