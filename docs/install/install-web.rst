
============
Installation
============

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Menu

   enable-login
   enable-remote-access
   attach-slurm
   troubleshooting

Supported operating systems
---------------------------

``nextPYP`` has been tested and works on the following operating systems:

 * `Rocky Linux 8.6 <https://docs.rockylinux.org/release_notes/8_6>`_
 * `CentOS 7 <https://wiki.centos.org/action/show/Manuals/ReleaseNotes/CentOS7.2009>`_
 * `Ubuntu 22.04.1 LTS <https://releases.ubuntu.com/22.04/>`_


Step 1: Prerequisites for installation
--------------------------------------

 * Service account:
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

 * Access to GPUs (optional):
    GPUs are only required to execute certain processing steps in ``nextPYP``. For example, they are needed to train neural networks used for particle picking (both for single-particle and tomography). ``nextPYP`` uses Apptainer_ which natively supports NVIDIA CUDA & AMD ROCm, but we have only tested it using NVIDIA GPUs with CUDA version 11.8. Support for other CUDA versions or AMD ROCm may require rebuilding the containers with driver and library versions matching the host configuration.


Step 2: Install operating system packages
-----------------------------------------

The only packages needed are Apptainer_ (formerly Singularity) and ``wget``. Instructions for installing
then vary by operating system:

.. _Apptainer: http://apptainer.org/

.. tabbed:: RedHat-based Linux (including CentOS and Rocky Linux)

  Before installing the packages, you will need first to enable the EPEL_ repository,
  if it was not enabled already:

  .. _EPEL: https://www.redhat.com/en/blog/whats-epel-and-how-do-i-use-it

  .. code-block:: bash

    sudo dnf install -y epel-release

  Then you can install the packages:

  .. code-block:: bash

    sudo dnf install -y apptainer wget

.. tabbed:: Ubuntu (22.04)

  Install `wget`:

  .. code-block:: bash

	  sudo apt-get install -y wget

  Download debian package for Apptainer:

  .. code-block:: bash

    wget https://github.com/apptainer/apptainer/releases/download/v1.1.0-rc.2/apptainer_1.1.0-rc.2_amd64.deb

  Install Apptainer:

  .. code-block:: bash

    sudo apt-get install -y ./apptainer_1.1.0-rc.2_amd64.deb


Step 3: Download and run the installation script
------------------------------------------------

First, create the folder where ``nextPYP`` will be installed. Something like ``/opt/nextPYP`` works well.
Then navigate to the folder in a shell session:

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
it does what you want, mark it executable and run it with administrator privileges.
You'll need to supply the name of the service account as the ``$PYP_USER`` environment variable.

.. code-block:: bash

  sudo chmod u+x install
  sudo PYP_USER=nextpyp ./install

If the installer gives an error like ``$username is not a valid group``, then you'll
need to set the group for the service account too, using the ``$PYP_GROUP`` environment variable:

.. code-block:: bash

  sudo PYP_USER=nextpyp PYP_GROUP=services ./install

The install script will download the rest of the needed software components and set them up.
Assuming fast download speeds, the installation script should finish in a few minutes.


Step 4: Check installation results
----------------------------------

Among other things, the installer created a `systemd` deamon named ``nextPYP`` to start and stop the
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

You can also access the website directly from the shell:

.. code-block:: bash

  wget http://localhost:8080 -O -

Running the ``wget`` command above should return a response like the following.

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

     This location should have fast read/write speeds, ideally in local storage.

     This is set to the system temporary directory by default, which is usually a safe starting point.
     But if you run out of space there, you can change this to a location with more space.

 * ``pyp.binds``
     Since ``PYP`` runs inside of a Singularity/Apptainer container, by default, no files from outside
     of the container will be visible to ``PYP``. To make files visible to ``PYP``, bind the directories
     containing those files into the container. Make those directories are also readable by the service account.

 * ``web.local``
     Directory for storing the database and user data.

     This location should have fast read/write speeds, ideally in local storage.
     It also will need many GiB (or even TiB) of space available.
     Make sure this directory is readable and writable by the service account.

     The default location for this folder is in the installation folder (typically ``/opt/nextPYP``),
     but if the OS filesystem is not very large, you should consider moving the local folder
     to a filesystem with more available space.

     When changing this setting, be sure to move existing local files to the new location.

Here is an example of how to specify these options in the configuration file:

.. code-block:: toml

  [pyp]
  scratch = '/scratch/nextPYP'
  binds = [ '/nfs', '/cifs' ]

  [web]
  local = '/bigspace/nextpyp/local'

After making changes to your configuration file, restart the application:

.. code-block::

  sudo systemctl restart nextPYP

There are many other configuration options supported beyond the ones described here. See the :doc:`full documentation for the configuration file<../reference/config>` for details.

Hopefully the services will start up perfectly and you can start using ``nextPYP`` right away. If not, there are a few useful places to look for debugging information. See :doc:`troubleshooting<./troubleshooting>` for more details.

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

  If you need to allow multiple different people to use the application, but want them to have
  separate projects and storage locations, follow these instructions to set up multi-user mode.

  * :doc:`Attach a SLURM cluster <./attach-slurm>`

  For large processing jobs, using a compute cluster can speed up results significantly.
  These instructions show how to attach a SLURM cluster to your installation.
