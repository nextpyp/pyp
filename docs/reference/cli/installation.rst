============
Installation
============

Supported operating systems
---------------------------

``PYP`` has been tested and works on the following operating systems:

 * `Rocky Linux 8.6 <https://docs.rockylinux.org/release_notes/8_6>`_
 * `CentOS 7 <https://wiki.centos.org/action/show/Manuals/ReleaseNotes/CentOS7.2009>`_
 * `Ubuntu 22.04.1 LTS <https://releases.ubuntu.com/22.04/>`_

The ``PYP`` command line interface only works when a SLURM cluster is attached (i.e., it does not work in standalone mode).

Step 1: Prerequisites for installation on a cluster
---------------------------------------------------

 * SLURM Cluster:
     ``PYP`` uses a SLURM_ compute cluster to do the data processing. The login node of the SLURM
     cluster must be reachable on the network from the machine where ``PYP`` will be installed.

 * Shared filesystem:
     ``PYP`` requires that the web server and the SLURM cluster share a single filesystem (e.g.
     an NFS storage system) and it be mounted at the same mount point on every machine.
     For example, if the shared filesystem is mounted on the SLURM cluster nodes as ``/nfs/data``,
     then those files should also be available on the web server machine as ``/nfs/data``.

 * Paswordless SSH access to the SLURM login node:
     The service account needs to have login access from the web server to the SLURM node via SSH without a password. This will require installing the public SSH key for the service account into the login system for the SLURM node. For a stock linux installation of sshd, that usually means copying the public key into a file like `/home/account/.ssh/authorized_keys`. But for SLURM clusters with a networked login system or SSO, you'll need to consult your organization's IT staff for SSH key installation instructions.

.. _SLURM: https://slurm.schedmd.com/overview.html

Step 2: (if needed) Install operating system packages
-------------------------------------------------

The only packages needed are Apptainer_ (formerly Singularity) and ``wget``.

.. _Apptainer: http://apptainer.org/

You can verify if these are installed in your system using:

  .. code-block:: bash

    command -v wget
    command -v apptainer

If they are not, you will need admin privileges to install them. Installation instructions vary by operating system:

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

First, create the folder where ``PYP`` will be installed. This folder must be on the shared
filesystem (unless you are installing in Standalone mode). If you mounted the shared filesystem at e.g. ``/nfs/data/``, then create a folder
called something like ``/nfs/data/apps/PYP``.

Then, navigate to the app folder you just created.

.. code-block:: bash

  cd "/nfs/data/apps/PYP"

Then, choose which version of ``PYP`` you want to install.

Then, download the installation script for that version. Assuming you chose "latest", run:

.. code-block:: bash

  wget https://nextpyp.app/files/pyp/latest/install-cli
  chmod u+x install-cli

The next step is to run the installer.

.. code-block:: bash

  ./install-cli

After the installation finishes (it should not take more than a couple of minutes), ``PYP`` is almost ready to use.
All that's left is to confirm (or modify) the configuration file.


Step 4: Review configuration
----------------------------

The installer created a default configuration file at ``config.toml`` in the installation folder.
This file is written in the TOML_ format.

.. _TOML: https://toml.io/en/

The installer did its best to guess the correct configuration options for your environment, but
almost certainly there are some settings that need review.

In particular, the ``slurm.host`` setting should be the hostname or address of the SLURM login node. If you happened to
install ``PYP`` on the SLURM login node, then congratulations! This setting is correct for you.
If ``PYP`` is installed on another machine instead, then be sure to correct the value to the real
SLURM login node.

Feel free to review any other configuration settings as well. The default configuration file has
a few comments to describe the settings configured there, but you can find more information in the
`full documentation for the configuration file <../config.html>`_.

.. note::

  To run ``PYP`` in Standalone mode, make sure there ``[slurm]`` section in the configuration file is removed.

Step 5 (recommended): Configure access system resources
-------------------------------------------------------

Configure how to access system resources by specifying the following parameters:

 * ``pyp.scratch``
     Directory for large (multi-GB) temporary files on the compute nodes. This location should have fast read/write speeds, ideally in local storage.

 * ``pyp.binds``
     Since ``PYP`` runs inside of a Singularity/Apptainer container, by default, no files from outside of the container will be visible to ``PYP``. To make files visible to ``PYP``, bind the directories containing those files into the container.

 * ``slurm.path`` (SLURM mode only)
     Path to the SLURM binaries on the login node.

 * ``slurm.queues`` (SLURM mode only)
     The names of any SLURM partitions to which users can submit ``PYP`` jobs.

 * ``slurm.gpuQueues`` (SLURM mode only)
     The names of any SLURM partitions with GPU hardware to which users can submit ``PYP`` jobs.

Here is an example of how to specify these options in the configuration file:

.. code-block:: toml

  [pyp]

  scratch = '/scratch/nextPYP'
  binds = [ '/nfs', '/cifs' ]

  [slurm]

  path = '/opt/slurm/bin'
  queues = [ 'general', 'quick' ]
  gpuQueue = [ 'gpu' ]


Step 6: Add ``PYP`` to your shell
---------------------------------

Add the following code to your shell configuration file (e.g., ``.bashrc`` or ``.bash_profile`` if using ``bash``):

.. code-block:: bash

    export PATH=$PATH:/nfs/data/apps/PYP
    export PYP_CONFIG=/nfs/data/apps/PYP/config.toml

Restart your shell for the changes to take effect.

If everything went well, you should be able to execute: ``pyp -h``.