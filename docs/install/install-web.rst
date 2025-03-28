
========
Overview
========

Supported operating systems
---------------------------

``nextPYP`` has been tested and works on the following operating systems:

* `Rocky Linux 8.6 <https://docs.rockylinux.org/release_notes/8_6>`_
* `CentOS 7 <https://wiki.centos.org/action/show/Manuals/ReleaseNotes/CentOS7.2009>`_
* `Ubuntu 22.04.1 LTS <https://releases.ubuntu.com/22.04/>`_
* `Debian 12 Bookworm <https://www.debian.org/releases/bookworm>`_

These instructions refer to the installation of ``nextPYP``'s' web interface. If you want to install the ``PYP`` command line interface, follow :doc:`these instructions<../cli/installation>` instead.

Step 1: Prerequisites for installation
--------------------------------------

* **Service account**
  
  ``nextPYP`` requires a service account to run the web server process.
  Because this user account runs a web server which may be exposed to external networks,
  like the public internet, the service account should not have administrative privileges.

  .. note::

    If you intend to attach a SLURM cluster later on, this account should have access
    to the required network resources, like the SLURM login node, and a shared filesystem.

  If you don't have a service account ready already, and you only need it to work on the local machine,
  you can create one in Debian-like Linux with:

  .. code-block:: bash

      sudo adduser --system --group nextpyp

  If you're using RHEL-like Linux, try:

  .. code-block:: bash

      sudo adduser --system --user-group nextpyp

* **Access to GPUs (optional)**
  
  GPUs are only required to execute certain processing steps in ``nextPYP`` (e.g.: train neural networks for particle picking, run MotionCor3 for frame alignment or Aretomo for tilt-series alignment and reconstruction). ``nextPYP`` uses Apptainer_ which natively supports NVIDIA CUDA & AMD ROCm, but we have only tested it with NVIDIA GPUs. If you have problems using AMD ROCm GPUs, please contact us.


Step 2: Install operating system packages
-----------------------------------------

The only packages needed are Apptainer_ (formerly Singularity) and ``wget``. Instructions for installing
then vary by operating system:

.. _Apptainer: http://apptainer.org/

.. comment:
   Looks like we're using sphinx-design for panels now?
   The panels in sphinx-design seem to be a bit different than panels from our old lib, sphinx-panels.
   See: https://sphinx-design.readthedocs.io/en/pydata-theme/tabs.html

.. tab-set::
  :class: custom-tab-set-style
  :sync-group: install_web_os

  .. tab-item:: RedHat-based Linux (including CentOS and Rocky Linux)

    Before installing the packages, you will need first to enable the EPEL_ repository,
    if it was not enabled already:

    .. _EPEL: https://www.redhat.com/en/blog/whats-epel-and-how-do-i-use-it

    .. code-block:: bash

      sudo dnf install -y epel-release

    Then you can install the packages:

    .. code-block:: bash

      sudo dnf install -y apptainer wget

  .. tab-item:: Debian-based Linux (including Ubuntu)

    Install `wget`:

    .. code-block:: bash

      sudo apt-get install -y wget

    Download debian package for Apptainer:

    .. code-block:: bash

      wget https://github.com/apptainer/apptainer/releases/download/v1.3.4/apptainer_1.3.4_amd64.deb

    Install Apptainer:

    .. code-block:: bash

      sudo dpkg -i apptainer_1.3.4_amd64.deb


Step 3: Download and run the installation script
------------------------------------------------

.. tab-set::
  :class: custom-tab-set-style
  :sync-group: install_web_user

  .. tab-item:: I'm using a regular user account
    :sync: user

    First, create the folder where ``nextPYP`` will be installed. The location can be anywhere you have write access, for example, ``~/nextPYP`` works well:

    .. code-block:: bash

      cd ~/
      mkdir nextPYP
      cd nextPYP

    Then, download the installation script:

    .. code-block:: bash

      wget https://nextpyp.app/files/pyp/latest/install

    Feel free to inspect the installation script. It's fairly simple. Once you're confident that
    it does what you want, mark it executable:

    .. code-block:: bash

      chmod u+x install

    Finally, run the installation script to install ``nextPYP``:

    .. code-block:: bash

      ./install


  .. tab-item:: I'm using an administrator account
    :sync: admin

    First, create the folder where ``nextPYP`` will be installed. This folder should be on the local
    filesystem of the web server machine. Something like ``/opt/nextPYP`` works well.
    This folder should be owned by `root` or your administrator account.
    The installation folder should *not* be owned by the service account, for security reasons.

    Navigate to the folder in a shell session:

    .. code-block:: bash

      sudo mkdir -p /opt/nextPYP
      cd /opt/nextPYP

    Then, download the installation script:

    .. code-block:: bash

      sudo wget https://nextpyp.app/files/pyp/latest/install

    .. note::

      Other versions can be installed by downloading an installation script by its version number.
      If you wanted to specifically install version ``0.5.0``, you would download the installation script at
      ``https://nextpyp.app/files/pyp/0.5.0/install``.

    Feel free to inspect the installation script. It's fairly simple. Once you're confident that
    it does what you want, mark it executable:

    .. code-block:: bash

      sudo chmod u+x install

    The installation script has a few different options, to handle different environments.
    In privileged installation, you'll need at least the ``PYP_USER`` option, and maybe some others too.
    All of the options are described below.

    * ``PYP_USER``
        The name of the service account. The service account should be an unprivileged user for security reasons.

    * ``PYP_GROUP``
        The group of the service account. By default, the installer will try using a group with the same name as the account. If the installer fails with an error like: ``$username is not a valid group``, then you'll need to set ``PYP_GROUP`` explicitly: eg, ``PYP_GROUP=services``

    * ``PYP_LOCAL``
        If your web server has access to fast local storage that is different than the storage used by the operating system (eg. NVMe SSDs mounted at ``/scratch``), this option will configure ``nextPYP`` to use it. Omitting this option will use a location inside the install folder for local storage instead.
        This setting should be the path to a folder on the local filesystem that is owned by the service account, eg. ``PYP_LOCAL="/media/nvme/nextPYP"``

    If you're installing onto a compute cluster with a shared filesystem, you'll need both the ``PYP_SHARED_DATA`` and ``PYP_SHARED_EXEC`` options:

    * ``PYP_SHARED_DATA``
        This option configures the shared location for run-time data created by ``nextPYP``. This folder should be owned by the service account and configured for read and write access, eg. ``PYP_SHARED_DATA="/nfs/users/service_acct/nextPYP/data"``

    * ``PYP_SHARED_EXEC``
        This option configures the shared location for executables and configuration. This folder should be owned by an adminisrator account and *not* the service account and configured for read-only access by the service account, eg. ``PYP_SHARED_EXEC="/nfs/users/service_acct/nextPYP/exec"``

    Choose the options and values according to your needs and then send them as environment variables to the installer.
    For example, if you were using only the service account option ``NEXT_PYP``, you would run the installer like this:

    .. code-block:: bash

      sudo PYP_USER="service_acct" ./install

    Or if you're doing a cluster installation, the install command might look like this:

    .. code-block:: bash

      sudo PYP_USER="service_acct" PYP_SHARED_DATA="/nfs/nextPYP/data" PYP_SHARED_EXEC="/nfs/nextPYP/exec" ./install

    .. note::

      Create any folders referenced by the installation options before running the installer.
      The installer will not create these folders for you.

The install script will download the rest of the needed software components and set them up.
Total download sizes are in the tens of gigabytes, so on a fast internet connection,
the installation script would need at least a few minutes to finish.


Step 4: Check installation results
----------------------------------

.. tab-set::
  :class: custom-tab-set-style
  :sync-group: install_web_user

  .. tab-item:: I'm using a regular user account
    :sync: user

    Now that ``nextPYP`` is installed, you can start the service and see if it works.

    To start the ``nextPYP`` website, run:

    .. code-block:: bash

      ./nextpyp start

    If the startup process is successful, your console should show a message similar to:

    .. code-block::

      Reading config.toml using CLI tool ...
      Host Processor started pid=1291 (/media/micromon/run/host-processor)
      Configuring environment ...
      Starting singularity container ...
      INFO:    instance started successfully

    To stop the ``nextPYP`` website, run:

    .. code-block:: bash

      ./nextpyp stop

  .. tab-item:: I'm using an administrator account
    :sync: admin

    Among other things, the installer created a ``systemd`` deamon named ``nextPYP`` to start and stop the
    application automatically. The daemon should be running now. Check it with:

    .. code-block:: bash

      sudo systemctl status nextPYP

    If all went well, you should be greeted with a response similar to the following.

    .. code-block::

      ● nextPYP.service - nextPYP
        Loaded: loaded (/usr/lib/systemd/system/nextPYP.service; enabled; vendor preset: disabled)
        Active: active (running) since Thu 2022-08-11 10:14:57 EDT; 4h 5min ago
      Main PID: 2774 (starter-suid)
          Tasks: 91 (limit: 23650)
        Memory: 708.3M
        CGroup: /system.slice/nextPYP.service
                ├─2774 Singularity instance: nextpyp [nextPYP]
                ├─2775 sinit
                ├─2793 /bin/sh /.singularity.d/startscript
                ├─2796 /bin/sh /opt/micromon/init.sh
                ├─2802 /usr/bin/python2 /usr/bin/supervisord -c /etc/supervisor/supervisord.conf
                ├─2893 /bin/sh /opt/micromon/bin/micromon.sh
                ├─2894 /usr/bin/mongod --config /tmp/mongod.conf
                └─2895 java -Xmx2048M @bin/classpath.txt io.ktor.server.netty.EngineMain


You can test that the ``nextPYP`` website is running directly from the shell:

.. code-block:: bash

  wget http://localhost:8080 -O -

Executing this command should return a response like the following:

.. code-block::

    --2023-11-15 11:46:35--  http://localhost:8080/
    Resolving localhost (localhost)... ::1, 127.0.0.1
    Connecting to localhost (localhost)|::1|:8080... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 353 [text/html]
    Saving to: ‘STDOUT’
    
    -                                    0%[                                                                 ]       0  --.-KB/s               <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>nextPYP</title>
        <link rel="icon" type="image/png" href="favicon.png">
        <script type="text/javascript" src="main.bundle.js"></script>
    </head>
    <body>
    <div id="mmapp"></div>
    </body>
    </html>
    -                                  100%[================================================================>]     353  --.-KB/s    in 0s      
    
    2023-11-15 11:46:35 (47.7 MB/s) - written to stdout [353/353]

If you get errors instead of something similar to the responses above, then the application did not start up successfully.
You can look for clues as to what went wrong by checking the various log files.
See :doc:`troubleshooting<./troubleshooting>` for more details.

If you're logged into the server locally (i.e., with a keyboard and a monitor), then you can visit the website
in your browser now at http://localhost:8080.

.. note::

  If you're logged into the server remotely over SSH, you won't be able to visit the website in your browser just yet.
  Remote network access to the website is disabled by default.
  To enable remote access, head to `Next steps`_.


Step 5 (recommended): Configure access to system resources
----------------------------------------------------------

The installer created a configuration file for you called ``config.toml`` in your installation folder.
This file is written in the TOML_ format.

.. _TOML: https://toml.io/en/

Configure how to access system resources by specifying the following parameters:

* ``pyp.scratch``
    Directory for large (multi-GB) temporary files used during computation.
    This location should have fast read/write speeds, ideally in local storage on the compute node.
    This is set to the system temporary directory by default, which is usually a safe starting point.
    But if you run out of space there, you can change this to a location with more space.

* ``pyp.binds``
    Since ``nextPYP`` runs inside an Apptainer container, by default, no files from outside
    of the container will be visible. To make them visible, you have to explicitly bind the directories
    containing those files into the container. Make sure those directories are also readable by the service account.

Here is an example of how to specify these options in the configuration file:

.. code-block:: toml

  [pyp]
  scratch = '/scratch/nextPYP'
  binds = [ '/nfs', '/cifs' ]

After making changes to your configuration file, restart the application:

.. tab-set::
  :class: custom-tab-set-style
  :sync-group: install_web_user

  .. tab-item:: I'm using a regular user account
    :sync: user

    .. code-block:: bash

      ./nextpyp stop
      ./nextpyp start

  .. tab-item:: I'm using an administrator account
    :sync: admin

    .. code-block:: bash

      sudo systemctl restart nextPYP

There are many other configuration options supported beyond the ones described here.
See the :doc:`full documentation for the configuration file<../reference/config>` for details.


Next steps
----------

You can start using the application right away. By default, it's installed in single-user mode,
runs computation jobs on the local server, and is only accessible locally. This is the simplest configuration
for the application, but you can enable other configurations using the linked instructions below.

* :doc:`Enable remote access<./enable-remote-access>`

  If you're not logged into the server locally (i.e., with a keyboard and monitor), then you'll need
  to enable remote access to use the website from the network. Follow these instructions to configure
  remote network access.

* :doc:`Enable multiple users <./enable-login>`

  If you need to allow different people to use the application, but want them to have
  separate projects and storage locations, follow these instructions to set up multi-user mode.

* :doc:`Attach a SLURM cluster <./attach-slurm>`

  For large processing jobs, using a compute cluster can speed up results significantly.
  These instructions show how to attach a SLURM cluster to your installation.
  If you installed ``nextPYP`` using the ``PYP_SHARED_DATA`` and ``PYP_SHARED_EXEC`` options,
  you'll want to follow this step to connect ``nextPYP`` to your SLURM cluster.


Upgrading to a new version
--------------------------

To upgrade to a new version, stop ``nextPYP`` and simply re-run the installation:

.. tab-set::
  :class: custom-tab-set-style
  :sync-group: install_web_user

  .. tab-item:: I'm using a regular user account
    :sync: user

    First, ``cd`` into the folder where you first installed ``nextPYP``.
    Then, stop the website, (re)run the installer, and then start the website again:

    .. code-block:: bash

      # stop nextPYP
      ./nextpyp stop

      # download the new installer and mark it executable
      sudo wget https://nextpyp.app/files/pyp/latest/install -O install
      sudo chmod u+x install

      # run the new installer to upgrade
      ./install

      # re-start nextPYP
      ./nextpyp start

  .. tab-item:: I'm using an administrator account
    :sync: admin

    .. code-block:: bash

      # stop nextPYP
      sudo systemctl stop nextPYP

      # stop the reverse proxy (only required if you configured remote access through untrusted networks)
      sudo systemctl stop nextPYP-rprox

      # download the new version's installer
      sudo wget https://nextpyp.app/files/pyp/latest/install -O install
      sudo chmod u+x install

      # re-run the installation
      # (be sure to use the same installation options you used the first time)
      sudo PYP_USER=nextpyp ./install

      # re-install the reverse proxy (only if you configured remote access through untrusted networks)
      sudo chmod u+x install-rprox
      sudo PYP_DOMAIN=myserver.myorganization.org ./install-rprox

    After the upgrade is complete, the installer will start the ``nextPYP`` daemon for you.

After this, you should be able to access the application the same way you did before the upgrade.


Getting Help
------------

Getting ``nextPYP`` installed and working correctly can be tricky sometimes,
especially since everyone's needs are just a little different.
We've done our best to build an install process that's flexible enough to work in many different environments,
but sometimes things still might not work out perfectly.

If you have questions, need clarification on any of the installation options, or are just looking for a little
help getting through the installation, don't hesitate to reach out on our `GitHub discussions <https://github.com/orgs/nextpyp/discussions>`_  board.
