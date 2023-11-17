.. _troubleshooting:

===============
Troubleshooting
===============

Hopefully the services will start up perfectly and you can start using ``nextPYP`` right away.
If not, there are a few useful places to look for debugging information.

Console output
~~~~~~~~~~~~~~

Console output for ``systemd`` daemons is saved in the systemd logs.
You can access the systemd logs using the ``journalctl`` command and the name of the service:

.. code-block:: bash

  sudo journalctl -u nextPYP

.. note::

  ``journalctl`` shows the oldest part of the logs by default, but if you want to see newest part where recent
  errors are most likely to appear, navigate to the end of the log using the page-down or end keys.


Log files
~~~~~~~~~

The various stages of service startup are written to log files in the ``local/logs`` sub-folder of your installation folder, eg ``/nfs/data/apps/nextPYP/local/logs``.

 * ``init.log``
     This log records the output of the application server apptainer container startup.
     It's the first process to run inside of the application server container and this log file should
     appear before any others.
     Errors here indicate that the apptainer container could not start successfully.

 * ``superd``
     This log records the output of ``supervisord``, the init system inside of the application server container.
     It runs after ``init.log`` and starts up the database and HTTP server procceses inside of the container.
     Errors here indicate that the database and HTTP servers may have failed to start.

 * ``mongod.log``
     This log records the output of the database, MongoDB. Errors here indicate that the database may be unable
     to operate successfully due to errors with the environment.

 * ``hostprocessor``
     This log records the output of the ``hostprocessor`` process, a small shell script to help the application
     server launch processes outside of the apptainer container on the host OS. The ``hostprocessor`` is
     used by the application server to run compute jobs when no SLURM cluster is attached.

 * ``micromon``
     This log records the output of the HTTP server and the application itself. Every time the application is
     started, it will print useful diagnostic information to the log. This information can help verify
     that configuration values are being applied correctly. Errors here can indicate that the HTTP server
     and application failed to start, and that certain requests to the application resulted in server-side errors.

     This log file is typically the last one to appear in the startup sequence. Its absence usually indicates
     that some earlier error (hopefully in one of the above logs) prevented the startup sequence from reaching
     this stage.
