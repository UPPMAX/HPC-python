.. meta::
   :description: Using packages
   :keywords: packages, modules, day 2

.. _use-packages:
Using packages
==============

.. admonition:: Learning outcomes

    - Practice using the documentation of your HPC cluster
    - Can find and load a Python package module
    - Can determine if a Python package is installed

Why Python packages are important
---------------------------------

Python packages are pieces of tested Python code.
Prefer using a Python package over writing your own code.

Why software modules are important
----------------------------------

Software modules allows users of any HPC cluster
to activate their favorite software of any version.
This helps to assure reproducible research.

Exercises
---------

Exercise 1: using Python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- login to your HPC cluster

.. admonition:: Forgot how to do this?
    :class: dropdown

    Answer can be found at
    `day 1 <https://uppmax.github.io/naiss_intro_python/sessions/using_the_python_interpreter/#exercise-1-login-to-your-hpc-cluster>`_

- load the Python module of the version below

+------------+----------------+
| HPC cluster| Python version |
+============+================+
| Alvis      | ``3.12.3``     |module load Python/3.12.3-GCCcore-13.3.0
+------------+----------------+
| Bianca     | ``3.11.4``     |
+------------+----------------+
| COSMOS     | ``3.11.5``     |
+------------+----------------+
| Dardel     | ``3.11.4``     |
+------------+----------------+
| Kebnekaise | ``3.11.3``     |
+------------+----------------+
| LUMI       | ``TBA``        |
+------------+----------------+
| Rackham    | ``3.12.7``     |
+------------+----------------+
| Tetralith  | ``3.10.4``     |
+------------+----------------+

.. admonition:: Forgot how to do this?
    :class: dropdown

    Answer can be found at
    `day 1 <https://uppmax.github.io/naiss_intro_python/sessions/using_the_python_interpreter/#exercise-2-load-the-python-module>`_

    .. note to self

        HPC Cluster|Link to documentation                                                                              |Solution
        -----------|---------------------------------------------------------------------------------------------------|------------------------------------------------------
        Alvis      |[short](https://www.c3se.chalmers.se/documentation/module_system/python_example/) or [long](https://www.c3se.chalmers.se/documentation/module_system/modules/) |`module load Python/3.12.3-GCCcore-13.3.0`
        Bianca     |[here](https://docs.uppmax.uu.se/software/python/#loading-python)                                  |`module load python/3.11.4`
        COSMOS     |[here](https://lunarc-documentation.readthedocs.io/en/latest/guides/applications/Python/)          |`module load GCCcore/13.2.0 Python/3.11.5`
        Dardel     |:warning: [here](https://support.pdc.kth.se/doc/software/module/) and [here](https://support.pdc.kth.se/doc/applications/python/)    |`module load bioinfo-tools python/3.11.4`
        Kebnekaise |[here](https://docs.hpc2n.umu.se/software/userinstalls/#python__packages)                          |`module load GCC/12.3.0 Python/3.11.3`
        LUMI       |:warning: [here](https://docs.lumi-supercomputer.eu/software/installing/python/)                   |Unknown
        Rackham    |[here](http://docs.uppmax.uu.se/software/python/)                                                  |`module load python`
        Tetralith  |[here](https://www.nsc.liu.se/software/python/)                                                    |`module load Python/3.10.4-env-hpc2-gcc-2022a-eb`



- Confirm that the Python package, indicated in the table below, is absent.
  You can use any way to do so.

+------------+----------------+
| HPC cluster| Python package |
+============+================+
| Alvis      | ``scipy``      |
+------------+----------------+
| COSMOS     | ?              |
+------------+----------------+
| Dardel     | ?              |
+------------+----------------+
| Kebnekaise | ?              |
+------------+----------------+
| LUMI       | ?              |
+------------+----------------+
| Rackham    | ?              |
+------------+----------------+
| Tetralith  | ?              |
+------------+----------------+

.. admonition:: Answer
    :class: dropdown

    On the terminal, type ``pip list`` to see all the
    packages that are installed.

    In all cases, the package is not yet installed,
    as that is what we'll be doing next :-)

    .. Alvis note to self

        [richelb@alvis1 ~]$ pip list
        Package           Version
        ----------------- -------
        flit_core         3.9.0
        packaging         24.0
        pip               24.0
        setuptools        70.0.0
        setuptools-scm    8.1.0
        tomli             2.0.1
        typing_extensions 4.11.0
        wheel             0.43.0

- Find the software module to load the package. Use either
  the documentation of the HPC center, or use the module system

.. admonition:: Answer: where is this documented?
    :class: dropdown

    +------------+---------------------------------------------------------------------------------------------+
    | HPC cluster| URL to documentation                                                                        |
    +============+=============================================================================================+
    | Alvis      | `Here <https://www.c3se.chalmers.se/documentation/module_system/python/#numpy-and-scipy>`__ |
    +------------+---------------------------------------------------------------------------------------------+
    | COSMOS     | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+
    | Dardel     | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+
    | Kebnekaise | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+
    | LUMI       | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+
    | Rackham    | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+
    | Tetralith  | ?                                                                                           |
    +------------+---------------------------------------------------------------------------------------------+

.. admonition:: Answer: how to use the module system?
    :class: dropdown

    In the terminal, type the following command:

    +------------+----------------------+
    | HPC cluster| Command              |
    +============+======================+
    | Alvis      | ``module spider SciPy``
    +------------+----------------------+
    | COSMOS     | ``module spider ?``  |
    +------------+----------------------+
    | Dardel     | ``module spider ?``  |
    +------------+----------------------+
    | Kebnekaise | ``module spider ?``  |
    +------------+----------------------+
    | LUMI       | ``?``                |
    +------------+----------------------+
    | Rackham    | ``module spider ?``  |
    +------------+----------------------+
    | Tetralith  | ``module spider ?``  |
    +------------+----------------------+

    .. Alvis note to self

          Python-bundle-PyPI:
        ---------------------------------------------------------------------------------------
            Description:
              Bundle of Python packages from PyPI

             Versions:
                Python-bundle-PyPI/2023.06-GCCcore-12.3.0
                Python-bundle-PyPI/2023.10-GCCcore-13.2.0
                Python-bundle-PyPI/2024.06-GCCcore-13.3.0


- Load the software module

.. admonition:: Answer
    :class: dropdown

    In the terminal, type the following command:

    +------------+-----------------------------------------------------------+
    | HPC cluster| Command                                                   |
    +============+===========================================================+
    | Alvis      | ``module load SciPy-bundle/2024.05-gfbf-2024a``           |
    +------------+-----------------------------------------------------------+
    | COSMOS     | ``module load ?``                                         |
    +------------+-----------------------------------------------------------+
    | Dardel     | ``module load ?``                                         |
    +------------+-----------------------------------------------------------+
    | Kebnekaise | ``module load ?``                                         |
    +------------+-----------------------------------------------------------+
    | LUMI       | ``?``                                                     |
    +------------+-----------------------------------------------------------+
    | Rackham    | ``module load ?``                                         |
    +------------+-----------------------------------------------------------+
    | Tetralith  | ``module load ?``                                         |
    +------------+-----------------------------------------------------------+

- See the package is now present

.. admonition:: Answer
    :class: dropdown

    From the terminal, type ``pip list`` to see all the
    packages that are installed.

    In all cases, the package is now installed.
    Well done!

.. Alvis note to self

    [richelb@alvis1 ~]$ module load SciPy-bundle/2024.05-gfbf-2024a
    [richelb@alvis1 ~]$ pip list
    Package                           Version
    --------------------------------- -----------
    alabaster                         0.7.16
    appdirs                           1.4.4
    asn1crypto                        1.5.1
    atomicwrites                      1.4.1
    attrs                             23.2.0
    Babel                             2.15.0
    backports.entry_points_selectable 1.3.0
    backports.functools_lru_cache     2.0.0
    beniget                           0.4.1
    bitarray                          2.9.2
    bitstring                         4.2.3
    blist                             1.3.6
    Bottleneck                        1.3.8
    CacheControl                      0.14.0
    cachy                             0.3.0
    certifi                           2024.6.2
    cffi                              1.16.0
    chardet                           5.2.0
    charset-normalizer                3.3.2
    cleo                              2.1.0
    click                             8.1.7
    cloudpickle                       3.0.0
    colorama                          0.4.6
    commonmark                        0.9.1
    crashtest                         0.4.1
    cryptography                      42.0.8
    deap                              1.4.1
    decorator                         5.1.1
    distlib                           0.3.8
    distro                            1.9.0
    docopt                            0.6.2
    docutils                          0.21.2
    doit                              0.36.0
    dulwich                           0.22.1
    ecdsa                             0.19.0
    editables                         0.5
    exceptiongroup                    1.2.1
    execnet                           2.1.1
    filelock                          3.15.1
    flit_core                         3.9.0
    fsspec                            2024.6.0
    future                            1.0.0
    gast                              0.5.4
    glob2                             0.7
    html5lib                          1.1
    idna                              3.7
    imagesize                         1.4.1
    importlib_metadata                7.1.0
    importlib_resources               6.4.0
    iniconfig                         2.0.0
    intervaltree                      3.1.0
    intreehooks                       1.0
    ipaddress                         1.0.23
    jaraco.classes                    3.4.0
    jaraco.context                    5.3.0
    jeepney                           0.8.0
    Jinja2                            3.1.4
    joblib                            1.4.2
    jsonschema                        4.22.0
    jsonschema-specifications         2023.12.1
    keyring                           24.3.1
    keyrings.alt                      5.0.1
    liac-arff                         2.5.0
    lockfile                          0.12.2
    markdown-it-py                    3.0.0
    MarkupSafe                        2.1.5
    mdurl                             0.1.2
    mock                              5.1.0
    more-itertools                    10.3.0
    mpmath                            1.3.0
    msgpack                           1.0.8
    netaddr                           1.3.0
    netifaces                         0.11.0
    numexpr                           2.10.0
    numpy                             1.26.4
    packaging                         24.1
    pandas                            2.2.2
    pastel                            0.2.1
    pathlib2                          2.3.7.post1
    pathspec                          0.12.1
    pbr                               6.0.0
    pexpect                           4.9.0
    pip                               24.0
    pkginfo                           1.11.1
    platformdirs                      4.2.2
    pluggy                            1.5.0
    ply                               3.11
    pooch                             1.8.2
    psutil                            5.9.8
    ptyprocess                        0.7.0
    py                                1.11.0
    py_expression_eval                0.3.14
    pyasn1                            0.6.0
    pybind11                          2.12.0
    pycparser                         2.22
    pycryptodome                      3.20.0
    pydevtool                         0.3.0
    Pygments                          2.18.0
    pylev                             1.4.0
    PyNaCl                            1.5.0
    pyparsing                         3.1.2
    pyrsistent                        0.20.0
    pytest                            8.2.2
    pytest-xdist                      3.6.1
    python-dateutil                   2.9.0.post0
    pythran                           0.16.1
    pytoml                            0.1.21
    pytz                              2024.1
    rapidfuzz                         3.9.3
    referencing                       0.35.1
    regex                             2024.5.15
    requests                          2.32.3
    requests-toolbelt                 1.0.0
    rich                              13.7.1
    rich-click                        1.8.3
    rpds-py                           0.18.1
    scandir                           1.10.0
    scipy                             1.13.1
    SecretStorage                     3.3.3
    semantic_version                  2.10.0
    setuptools                        70.0.0
    setuptools-scm                    8.1.0
    shellingham                       1.5.4
    simplegeneric                     0.8.1
    simplejson                        3.19.2
    six                               1.16.0
    snowballstemmer                   2.2.0
    sortedcontainers                  2.4.0
    Sphinx                            7.3.7
    sphinx-bootstrap-theme            0.8.1
    sphinxcontrib-applehelp           1.0.8
    sphinxcontrib-devhelp             1.0.6
    sphinxcontrib-htmlhelp            2.0.5
    sphinxcontrib-jsmath              1.0.1
    sphinxcontrib-qthelp              1.0.7
    sphinxcontrib-serializinghtml     1.1.10
    sphinxcontrib-websupport          1.2.7
    tabulate                          0.9.0
    threadpoolctl                     3.5.0
    toml                              0.10.2
    tomli                             2.0.1
    tomli_w                           1.0.0
    tomlkit                           0.12.5
    typing_extensions                 4.11.0
    tzdata                            2024.1
    ujson                             5.10.0
    urllib3                           2.2.1
    versioneer                        0.29
    virtualenv                        20.26.2
    wcwidth                           0.2.13
    webencodings                      0.5.1
    wheel                             0.43.0
    xlrd                              2.0.1
    zipfile36                         0.1.3
    zipp                              3.19.2




