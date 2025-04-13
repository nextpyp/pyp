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


Frequently Asked Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~

I get the following error message in the console when trying to start nextPYP. What does it mean and how do I fix it?
---------------------------------------------------------------------------------------------------------------------

 | ERROR  : Could not write info to setgroups: Permission denied
 | ERROR  : Error while waiting event for user namespace mappings: no event received

This error is usually caused by an incomplete installation of `apptainer`_, the container runtime used by nextPYP.
On some systems, apptainer needs to be granted additional permissions with kernel security modules, like `AppArmor`_.

.. _apptainer: https://apptainer.org/
.. _AppArmor: https://apparmor.net/

To check if AppArmor is running on your system, run:

.. code-block:: bash

  sudo systemctl status apparmor

If the above command indicates no apparmor service is running (or even installed), then some security module
other than AppArmor may be interfering with apptainer, but we don't know how fix this issue in that case.
If the above command does indicate an active AppArmor service,
then you can grant the necessary permissions to apptainer by creating an AppArmor profile.

Create a file at ``/etc/apparmor.d/apptainer`` with the following content:

 | # Permit unprivileged user namespace creation for apptainer starter
 | abi <abi/4.0>,
 | include <tunables/global>
 | profile apptainer /usr/libexec/apptainer/bin/starter{,-suid}
 | flags=(unconfined) {
 |   userns,
 |   # Site-specific additions and overrides. See local/README for details.
 |   include if exists <local/apptainer>
 | }

Double-check that the ``/usr/libexec/apptainer/bin/starter`` path above refers to your actual
apptainer executable:

.. code-block:: bash

  ls -al "/usr/libexec/apptainer/bin/starter"

If the executable file does exist at that location, then the above AppArmor profile should work on your system.
If no executable file exists at that location, then you'll have to try to find the correct location.
Try a search command like:

.. code-block:: bash

  find /usr -wholename "*/apptainer/bin/starter"

Once you've found the correct location of the apptainer executable file, edit the path above in the AppArmor profile.

Finally, to apply the new AppArmor profile, ask the AppArmor service to reload its configuration with:

.. code-block:: bash

  sudo systemctl reload apparmor

You should now be able to start your nextPYP service without encountering the original error.

If you want to lean more, you can also `read the original apptainer docs`_ about how apptainer interacts with AppArmor.

.. _read the original apptainer docs: https://github.com/apptainer/apptainer/blob/main/INSTALL.md#apparmor-profile-ubuntu-2310
