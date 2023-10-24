============
Contributing
============


Python PEP8
===========
Generally, `PEP 8 <http://www.python.org/dev/peps/pep-0008/>`_ must be observed.

Git pre-commit hooks (`isort` for import sorting, and `black` for formatting) will aid you in getting the correct code style.

Tests
=====

Built-in tests are integrated into git hooks. Everytime `git push` is run, tests are triggered. Tests are written using the `pytest <https://pytest.org/>`_ framework and located in `tests/test_pyp.py`.

Feel free to add tests, especially if you find parts of PYP to be **extremely** buggy.

Documentation
=============

The docs are automatically pulled from comments under each module, class, method, or function.

When you contribute to the code, please include documentation. If you find pieces of undocumented code, you're very welcome to add to those too!

Use the `numpy <https://numpydoc.readthedocs.io/en/latest/format.html>`_ docstring format. There are also examples of documentation in the source code under `pyp/inout/metadata/frealign_parfile.py`, `pyp/system/wrapper_functions.py`, and `pyp/pyp_main.py`.

Logging
=======

Rather than using simple `print` statements, Using the built-in `Python Logging`_ module enables us to filter logs based on severity/importance levels and to redirect messages to files.

Currently, we are directing logs to standard I/O (printing to console) but creating separate streams to stdio and files is possible.

Exampes of logging are in ``src/pyp_main.py`` and ``src/frealign.py``. 

.. warning::
    The current logger uses pretty color output when you are running on a terminal. Unfortunately, this produces additional escape characters that will also be saved in the log files. Using programs like ``more`` will correctely interpret the escape characters, but most programs will not.

.. note::
    | It is advisable to use old-style Python string formatting for logging, 
    | e.g., ``logger.info("PBS_O_WORKDIR = %s", os.environ['PBS_O_WORKDIR'])``
    | instead of
    | ``logger.info(f"PBS_O_WORKDIR = {os.environ['PBS_O_WORKDIR']}")``
    | For more information: refer to `logging.debug function signature <https://docs.python.org/3/library/logging.html#logging.debug>`_

.. _Python Logging:
    https://docs.python.org/3/library/logging.html

IDE
===

We highly encourage VSCode. VSCode has a lot of extensions and built-in features that help you become more productive.

First, you can easily SSH into an HPC cluster and directly edit/run code on the cluster.

Pylance is a very versatile extension that provides autocomplete, helps you refactor code, and fades out unused variables, amongst other features.

Use Python Docstring Generator to generate numpydoc templates for all functions, classes, and modules.

You can also configure `Black`, `isort`, `flake8` within VSCode!

