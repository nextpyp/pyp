
========
Overview
========


.. info:: Upgrading from an older version of ``nextPYP``?

  Skip down to the :ref:`instructions for upgrades<upgrade>`. They're a bit different than first-time installations.


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
  you can create one with:

  .. code-block:: bash

    sudo useradd --system --user-group nextpyp

* **Storage space**

  Cryo-ET and cryo-EM data requires significant amounts of storage space -- on the order of terabytes.
  At the very least, you'll want to have access to a multi-terabyte partition (either an SSD mounted locally or a large capacity 
  network-mounted share). nextPYP does best with fast local storage, but it can use slower remote storage as well.
  If you configure nextPYP to store its data on a typically-sized 10s of gigabyte operating system partition,
  you'll very quickly run out of space.

  .. tip::

    You can quickly see what storage devices your system has and how much space is available
    by running the ``df`` command:

    .. code-block:: bash

      df -h

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
  :sync-group: install_web_os

  .. tab-item:: Debian-based Linux (including Ubuntu)
    :sync: debian

    Install `wget`:

    .. code-block:: bash

      sudo apt-get install -y wget

    Download debian package for Apptainer:

    .. code-block:: bash

      wget https://github.com/apptainer/apptainer/releases/download/v1.3.4/apptainer_1.3.4_amd64.deb

    Install Apptainer:

    .. code-block:: bash

      sudo dpkg -i apptainer_1.3.4_amd64.deb

  .. tab-item:: RedHat-based Linux (including CentOS and Rocky Linux)
    :sync: rhel

    Before installing the packages, you will need first to enable the EPEL_ repository,
    if it was not enabled already:

    .. _EPEL: https://www.redhat.com/en/blog/whats-epel-and-how-do-i-use-it

    .. code-block:: bash

      sudo dnf install -y epel-release

    Then you can install the packages:

    .. code-block:: bash

      sudo dnf install -y apptainer wget


Step 3: Download and run the installation script
------------------------------------------------

.. tab-set::
  :sync-group: install_web_user

  .. tab-item:: I'm using a regular user account
    :sync: user

    First, create the folder where ``nextPYP`` will be installed.
    The location can be anywhere you have write access and also has lots of free space available.
    You'll probably want at least one terabyte to start. Assuming you have a generous storage quota in
    your home folder, try ``~/nextPYP``:

    .. code-block:: bash

      cd ~/
      mkdir nextPYP
      cd nextPYP

    Then, download the installation script:

    .. code-block:: bash

      wget https://nextpyp.app/files/pyp/latest/install

    Feel free to inspect the installation script. It's meant to be fairly readable. Once you're confident that
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

    This folder won't need huge amounts of free space, but you'll need at least a few hundred megabytes or so
    for the executable files.

    .. warning::

      While you can install ``nextPYP`` to a networked folder, doing so often comes with performance penalties,
      since reading files from remote folders can be much slower than a local folder. For the best performance,
      install ``nextPYP`` to a folder in the web server's local filesystem. A good choice is ``/opt`` which is
      traditionally used in Linux for optional software.

    This folder should be owned by `root` or your administrator account.
    The installation folder should *not* be owned (or be writatble by) by the service account,
    for security reasons.

    Navigate to the folder in a shell session:

    .. code-block:: bash

      sudo mkdir -p /opt/nextPYP
      cd /opt/nextPYP

    Then, download the installation script:

    .. code-block:: bash

      sudo wget https://nextpyp.app/files/pyp/latest/install

    .. note::

      Other versions can be installed by downloading an installation script by its version number.
      In the URL above, replace ``latest`` with the desired version number.
      For example, if you wanted to specifically install version ``0.5.0``, you would download the
      installation script at ``https://nextpyp.app/files/pyp/0.5.0/install``.

    Feel free to inspect the installation script. It's meant to be fairly readable. Once you're confident that
    it does what you want, mark it executable:

    .. code-block:: bash

      sudo chmod u+x install

    The installation script has a few different settings, configured as environment variables, to handle different
    needs during installation. Choose the scenario below that describes your computing hardware to
    explain the settings you'll need for installation.

    .. tab-set::
      :sync-group: install_web_hardware

      .. tab-item:: Desktop workstation
        :sync: workstation

        To install on a typical workstation, you'll need to set the ``PYP_USER`` and ``PYP_STORAGE`` settings.

        * ``PYP_USER``
            The name of the service account that you created in the prerequisites section, probably ``nextpyp``.

        * ``PYP_STORAGE``
            This folder will be used to hold all of nextPYP's data files and requires a lot of storage space.
            Set this setting to a folder on storage device with at least a terabyte of capacity.
            Ideally, this storage device is a large-capacity SSD or hard drive that is attached directly to your
            workstation and mounted in the local filesystem.

            This folder should exist, but it should be empty before installation. The installer will create
            subfolders in this folder to hold different kinds of data.

            The folder path might look something like: ``/large-storage/nextpyp``.

            Finally, the folder should be owned by ``root``. If it doesn't exist already, you can create it with:

            .. code-block:: bash

              sudo mkdir -p "/large-storage/nextpyp"

        Once you've decided what values to use for these settings, run the installer like this:

        .. code-block:: bash

          sudo PYP_USER="service_acct" PYP_STORAGE="/large-storage/nextpyp" ./install

      .. tab-item:: Compute cluster
        :sync: cluster

        For a cluster installation, there are several required settings, and a few optional ones.
        They're all described in detail below.

        * ``PYP_USER`` (required)
            The name of the service account. The service account should be an unprivileged user for security reasons.
            This user should also have read and write access to any filesystems shared with the cluster.

        * ``PYP_GROUP`` (optional)
            The group of the service account. By default, the installer will try using a group with the same name as the
            account. If the installer fails with an error like: ``$username is not a valid group``, then you'll need to
            set ``PYP_GROUP`` explicitly: eg, ``PYP_GROUP=services``

        * ``PYP_LOCAL`` (optional)
            The local folder holds mainly the nextPYP database files, so it should be in fast local
            storage. A storage device like an NVME or an SSD is ideal here.

            Without this setting, the installer will place the local folder under the installation folder.
            If the storage device serving your installation folder has at least a hundred gigabytes of space,
            the default is probably fine.

            If not, then you'll want to set this setting to a folder with more space.
            In that case, set ``PYP_LOCAL`` to a folder that already exists and is owned
            by the service account, eg, ``PYP_LOCAL="/nvme/nextPYP"``.

        * ``PYP_SHARED_DATA`` (required)
            This folder holds all the data that is shared between the web server and the compute nodes in the cluster.
            Set this setting to a folder on your networked filesystem (e.g., NFS) that has lots of free space --
            at least a few terabytes. Over time, this folder can grow very large --
            potentially tens or hundreds of terabytes, or even more.

            This folder should already exist and by owned by the service account,
            eg, ``PYP_SHARED_DATA="/nfs/users/service_acct/nextPYP/data"``.

        * ``PYP_SHARED_EXEC`` (required)
            This folder holds executable files and configuration shared between the web server and the compute nodes.

            This folder should already exist and be owned by an administrator account, *not* the service account.
            The service account should have read-only access to this folder. For security, the service account must *not*
            have write access to the executable and configuration files here.

            Pick a folder on your networked filesystem that already exists and has at least a few tens of gigabytes
            of space, eg, ``PYP_SHARED_EXEC="/nfs/nextPYP/exec"``. The executable files stored here are container images
            which can get pretty big.

        * ``PYP_SCRATCH`` (required)
            This folder holds temporary data for computations on the compute nodes. It should be hosted on fast local
            storage devices like NVME drives or SSDs *on each compute node, not networked storage*.
            The web server has no need to access this folder.

            This folder should have hundreds of gigabytes of free space.

            .. warning::

              On many systems, ``/tmp`` may not be large enough. If you want to use ``/tmp`` as scratch,
              verify it has enough space first.

            This folder should already exist and be writable by the service account,
            eg, ``PYP_SCRATCH=/scratch/nextPYP``

        Choose the settings according to your needs and then send them as environment variables to the installer.
        For example, setting a couple of the settings for the installer would look like this:

        .. code-block:: bash

          sudo PYP_USER="service_acct" PYP_SHARED_DATA="/nfs/nextPYP/data" ./install

        .. note::

          Create any folders referenced by the installation settings before running the installer.
          The installer will not create these folders for you.

The install script will download the rest of the needed software components and set them up.
Total download sizes are in the tens of gigabytes, so on a fast internet connection,
the installation script would need at least a few minutes to finish.


Step 4: Check installation results
----------------------------------

.. tab-set::
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

If you're logged into the server locally (e.g., with a keyboard and a monitor or some kind of
remote desktop software like VNC), then you can visit the website in your browser now at http://localhost:8080.

.. note::

  If you're logged into the server remotely over SSH, you won't be able to visit the website in your browser just yet.
  Remote network access to the website is disabled by default.
  To enable remote access, head to `Next steps`_ after you've finished the rest of the numbered steps.


Step 5: Configure your data folders
-----------------------------------

nextPYP uses `containerization`_ technology to help keep the install process as simple as we can make it,
but that comes with some tradeoffs.
One tradeoff is that since containerized apps operate inside of an isolated virtual filesystem,
these apps can't see all of the files in your real filesystem by default.
Meaning, nextPYP won't be able to see your Cryo-EM/ET data by default either.

.. _containerization: https://en.wikipedia.org/wiki/Containerization_(computing)

To get nextPYP to see your data, you'll have to "bind" your data path(s) into the container's filesystem.
You can do this by adding your data folder paths to the nextPYP configuration file.

The installer created a configuration file for you called ``config.toml`` and there's a symlink to it in
your installation folder. The configuration file is written in a configuration language called TOML_.
TOML is pretty similar to JSON, if that's familiar to you, but TOML is a bit nicer to use for this kind of thing.

.. _TOML: https://toml.io/en/

To add (aka "bind") your data folders into nextPYP's container,
edit the ``config.toml`` file with your favorite text editor.
Under the ``[pyp]`` section of the configuration file, look for a line that looks like this:

.. code-block:: toml

    binds = []

In, TOML, ``[]`` is an empty array (or list), so by default the binds list is empty.
To bind your data folder(s), add the paths (as strings) to the list. That might look something like this:

.. code-block:: toml

    binds = ['/path/to/my/data']

Or this:

.. code-block:: toml

    binds = [
      '/big-storage/cryo-data',
      '/other-big-storage/cryo-data'
    ]

After making changes to your configuration file, restart the application to apply the changes:

.. tab-set::
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

There are many other configuration options beyond the ones described here.
See the :doc:`full documentation for the configuration file<../reference/config>`
to learn about all of the other configurable settings.


Next steps
----------

You can start using the application right away. By default, it's installed in single-user mode,
runs computation jobs on the local server, and is only accessible locally. This is the simplest configuration
for the application, but you can enable other configurations using the linked instructions below.

* :doc:`Enable remote access<./enable-remote-access>`

  If you're not logged into the server locally (e.g., with a keyboard and monitor or some kind of
  remote desktop software like VNC), then you'll need to enable remote access to use the website from the network.
  Follow these instructions to configure remote network access.

* :doc:`Enable multiple users <./enable-login>`

  If you need to allow different people to use the application, but want them to have
  separate projects and storage locations, follow these instructions to set up multi-user mode.

* :doc:`Attach a SLURM cluster <./attach-slurm>`

  For large processing jobs, using a compute cluster can speed up results significantly.
  These instructions show how to attach a SLURM cluster to your installation.
  If you installed ``nextPYP`` using the ``PYP_SHARED_DATA`` and ``PYP_SHARED_EXEC`` options,
  you'll want to follow this step to connect ``nextPYP`` to your SLURM cluster.


.. _upgrade:

Upgrading to a new version
--------------------------

Step 1: Pre-installation steps (conditional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

  **Upgrading from v0.6.5 (or earlier) to v0.7.0 (or later) ?**

  We made the installer a lot smarter starting with v0.7.0, but if you're upgrading from an older
  version than that, there are few manual steps you'll have to do to catch up.

  .. admonition:: Manual Steps
    :collapsible:

    .. tab-set::
      :sync-group: install_web_user

      .. tab-item:: I'm using a regular user account
        :sync: user

        No extra steps needed.

      .. tab-item:: I'm using an administrator account
        :sync: admin

        #. Create a folder for shared executables

           Starting with v0.7.0, nextPYP stores executable files that may need to be shared with cluster compute nodes
           in a separate folder from the web server executables, which don't need to be shared with cluster compute nodes.

           You'll need to do these steps manually even if you're not using a compute cluster, since the same folder
           structure is also used for standalone workstation computers.

           Before upgrading, you'll need to create a folder for these executable files and then configure the installer
           to use it. This folder should be owned by ``root`` or an administrator account. It should **not** be owned or
           be writable by the service account. The service account should have read-only access to these executable files.
           The executable files are on the order of tens of gigabytes in size, so make sure your folder choice has enough
           free space.

           After you've created the folder and set the appropriate ownership and permissions, configure the installer
           to use it during the upgrade by setting the ``PYP_SHARED_EXEC`` environment variable, for example:

           .. code-block:: bash

             PYP_SHARED_EXEC="/storage/nextPYP/sharedExec"

        #. Create symlinks for local and shared data folders, if needed

           If your ``local`` and ``shared`` folders exist directly inside of your installation folder, you can skip
           this step.

           But if your ``local`` or ``shared`` folders are anywhere else, you should create a symlink from those
           locations to folders directly inside your installation folder. The resulting symlinks inside your installation
           folder should be named ``local`` and ``shared`` respectively. You can find the location of your ``local``
           and ``shared`` folders by examining your ``config.toml`` file, in the ``web.localDir`` and ``web.sharedDir``
           settings.

           So, for example, if your ``local`` folder is at ``/network/nextPYP/local`` and your installation folder is at
           ``/opt/nextPYP``, then you'll make the symlink like this:

           .. code-block:: bash

             sudo ln -s "/network/nextPYP/local" "/opt/nextPYP/"

           And then do the same thing for your shared folder. After both folders are symlinked, the installation script
           can now auto-detect your existing folders.


Step 2: Run the installation script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To upgrade to a new version, stop ``nextPYP``, download the new installer, run it, and then re-start ``nextPYP``.

.. tab-set::
  :sync-group: install_web_user

  .. tab-item:: I'm using a regular user account
    :sync: user

    First, ``cd`` into the folder where you first installed ``nextPYP``.
    Then, stop the website, (re)run the installer, and then start the website again:

    .. code-block:: bash

      # stop nextPYP
      ./nextpyp stop

      # download the new installer and mark it executable
      wget https://nextpyp.app/files/pyp/latest/install -O install
      chmod u+x install

      # run the new installer to upgrade
      ./install

  .. tab-item:: I'm using an administrator account
    :sync: admin

    .. code-block:: bash

      # stop nextPYP
      sudo systemctl stop nextPYP

      # download the new version's installer
      sudo wget https://nextpyp.app/files/pyp/latest/install -O install
      sudo chmod u+x install

      # run the new install script
      # If upgrading from v0.6.5 or earlier, you will need to set the PYP_SHARED_EXEC variable, e.g.:
      # sudo PYP_SHARED_EXEC="/storage/nextPYP/sharedExec" ./install
      sudo ./install


Step 3: Post-installation steps (conditional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a general rule, the installer won't make any changes to your existing configuration file,
or delete any potentially important files. So, some upgrades may require you to take
these steps manually.

If manual steps are needed after an upgrade, you'll see a message like the following in your console:

.. code-block::

  ===============================================================================
  |  BUT WAIT! There's still a bit more you need to do                          |
  |-----------------------------------------------------------------------------|

See below for more information about manual upgrade steps that are specific to each version of nextPYP.


.. admonition:: Upgrading from v0.6.5 (or earlier) to v0.7.0 (or later) ?
  :collapsible:

  A rough outline of the steps you'll need to do are described below.

  #. Delete the old container file
  #. Remove the old container configuration from your ``config.toml`` file.
  #. Add the new folder location for the shared executables folder to your ``config.toml`` file.

  The installer's prompt will contain much more detailed information though, including the exact locations
  of the relevant files, and full commands needed to do some of the tasks that you can copy into your terminal.


Step 4: Start nextPYP again
~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the installer has finished, and you have completed any manual post-installation steps,
start nextPYP again:

.. tab-set::
  :sync-group: install_web_user

  .. tab-item:: I'm using a regular user account
    :sync: user

    .. code-block:: bash

      ./nextpyp start

  .. tab-item:: I'm using an administrator account
    :sync: admin

    If no post-installation steps were required, the installer should have already re-started nextPYP for you.
    You can check the status of the nextPYP daemon with:

    .. code-block:: bash

      systemctl status nextPYP

    If post-installation steps were required, after completing those steps, start nextPYP again with:

    .. code-block:: bash

      sudo systemctl start nextPYP


Getting Help
------------

Getting ``nextPYP`` installed and working correctly can be tricky sometimes,
especially since everyone's needs are just a little different.
We've done our best to build an install process that's flexible enough to work in many different environments,
but sometimes things still might not work out perfectly.

If you have questions, need clarification on any of the installation options, or are just looking for a little
help getting through the installation, visit the :doc:`Support<../known-issues>` page for a list of available support resources.
