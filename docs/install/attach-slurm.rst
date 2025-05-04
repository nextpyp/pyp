
======================
Attach a SLURM Cluster
======================

The :doc:`base installation instructions <./install-web>` install a complete working application that does
all the computation on the local machine.

If, after you've installed the base application, you want to improve performance by attaching a SLURM_ cluster,
these instructions will show you how.

.. _SLURM: https://slurm.schedmd.com/overview.html


Step 1: Prerequisites
---------------------

* **SLURM scheduler**
    ``nextPYP`` uses a SLURM_ compute cluster to do the data processing. The login node of the SLURM cluster must be reachable on the network from the machine where ``nextPYP`` is installed.

* **Shared filesystem**
    ``nextPYP`` requires that the web server and the SLURM cluster share a single filesystem (e.g.
    an NFS storage system) and it be mounted at the same mount point on every machine.
    For example, if the shared filesystem is mounted on the SLURM cluster nodes as ``/nfs/data``,
    then those files should also be available on the web server machine as ``/nfs/data``.
    Additionally, you should have installed ``nextPYP`` using the ``PYP_SHARED_DATA`` and ``PYP_SHARED_EXEC`` options.

* **Service account**
    ``nextPYP`` requires a service account to both run the web server process, and access files on
    the shared filesystem. This user should be the same on the web server machine and the SLURM cluster.
    Because this user account runs the web server on the web server machine (which may be exposed to
    the public internet), the service account should not have administrative privileges.

* **Passwordless SSH access to the SLURM login node**
    The service account needs to have login access from the web server to the SLURM node via SSH without a password.
    This will require installing the public SSH key for the service account into the login system for the SLURM node.
    For a stock linux installation of ``sshd``, that usually means copying the public key into a file like
    ``/home/account/.ssh/authorized_keys``, for example, using the command ``ssh-copy-id user@domain``. But for SLURM clusters with a networked login system or SSO,
    you'll need to consult your organization's IT staff for SSH key installation instructions.

* **Remote access enabled**
    To attach a SLURM cluster, you will first need to enable remote access for your web server using the
    :doc:`enable remote access <./enable-remote-access>` installation instructions.


Step 2: Configuration
---------------------

Website configuration
~~~~~~~~~~~~~~~~~~~~~

In the ``config.toml`` file in your installation folder, add (or change) the ``web.webhost`` setting.
This setting configures how jobs running on compute nodes in the cluster should connect to the web server
and should be set to the URL of ``nextPYP`` on the web server *from the point of view of the compute node*.

.. tip::

  The ``web.host`` and ``web.webhost`` settings are different, but it's easy to confuse them for one another.
  The ``web.host`` setting controls what interfaces the web server binds to, while the ``web.webhost`` setting
  controls what URL ``pyp`` should use to connect to the web server. With the full benefit of hindsight,
  the ``web.webhost`` setting might be better named ``web.url``, but alas, that would be a breaking change.

Depending on how your network is configured, choose one of the following options:

.. md-tab-set::

  .. md-tab-item:: I configured remote access using a private network

    If you have enabled remote access using a private network,
    then the correct value for ``web.webhost`` will look something like:

    .. code-block:: toml

      [web]
      webhost = 'http://nextpyp.internal.myorganization.org:8080'

    For private networks, the ``http`` protocol (not ``https``) should be used here.
    And the port should be specified in the url explicitly, eg ``:8080``.

    .. note::

      If you have changed the port from the default value using the ``web.port`` option, use that
      port number here instead of 8080.


  .. md-tab-item:: I configured remote access using a reverse proxy

    If you have configured remote access to the web server using the reverse proxy,
    then the correct value for ``web.webhost`` will look something like:

    .. code-block:: toml

      [web]
      webhost = 'https://nextpyp.myorganization.org'

    For access over public networks, the ``https`` protocol (not ``http``) should be used here.
    And the port (``443`` for ``https``) should not be explicitly specified in the URL,
    since the HTTPs protocol uses the correct port implicitly.


SLURM configuration
~~~~~~~~~~~~~~~~~~~

Then add a new ``[slurm]`` section to the config file as well.
At a minimum, we'll need to set the ``slurm.host`` and the ``slurm.templatesDir`` properties, for example:

.. code-block:: toml

  [slurm]
  host = 'slurm-login.myorganization.org'
  templatesDir = '/opt/nextpyp/templates'

Each of the required settings are described in more detail below.

#. ``slurm.host``

   The ``slurm.host`` property is the hostname of the SLURM login node, from the point of view of the ``nextPYP``
   website server. The ``nextPYP`` web server process will SSH into the SLURM login node using this hostname to
   submit jobs to SLURM.

#. ``slurm.templatesDir``

   The ``slurm.templatesDir`` setting configures the location of :doc:`SLURM submission templates<../reference/templates>`.
   You'll need to create this folder somewhere that the service account can read, but not write.
   This folder doesn't need to be shared with compute nodes, to a location on the local filesystem of the web server
   machine is a good choice. A subfolder of your installation folder that is owned by ``root`` or your administrator
   account works well, like in the example above.

   Once the folder is ready, download the default SLURM submission template into the folder:

   .. code-block:: bash

     wget https://nextpyp.app/files/pyp/latest/default.peb -O /opt/nextpyp/templates/default.peb

   .. note::

     If you're not using the latest version of ``nextPYP``, change the url of the ``default.peb`` file above
     to match the version you're using by replacing the ``latest`` path component with your version number. e.g,.
     ``https://nextpyp.app/files/pyp/0.7.0/default.peb``

   This template file allows you to configure how ``nextPYP`` submits jobs to SLURM, for example,
   by adding additional parameters that are specific to your computing environment. For more information about how
   to customize ``nextPYP`` using SLURM templates, see :doc:`SLURM submission templates<../reference/templates>`.

Feel free to add any other relevant SLURM configuration here as well. You can find more information about all of
the available settings in the :doc:`full documentation for the configuration file <../reference/config>`.

Some commonly-used options are described below:

.. admonition:: ``slurm.path``
  :collapsible:

  Path to the SLURM binaries folder on the login node, if they are not already searchable using the
  usual ``$PATH`` configuration. e.g., ``path = '/some/nonstandard/location/slurm/bin'``.

  If your slurm binaries are in a standard location, like ``/bin`` or ``/usr/bin`` or a folder in the ``$PATH``
  environment variable, then you won't need to set the path explicitly.

Once you've finished making changes to your configuration file, restart ``nextPYP``:

.. code-block:: bash

  sudo systemctl restart nextPYP


Step 3: SSH access
------------------

To process a compute job, the website will attempt to SSH into the login node of the SLURM cluster to submit jobs.
For this connection to work, the website must have access to an SSH key.

To generate a new SSH key for the service account, run the following commands as the service account:

.. code-block:: bash

  cd ~/.ssh
  ssh-keygen -t rsa -f id_rsa
  cat id_rsa.pub >> authorized_keys
  chmod go-w authorized_keys

.. tip::

  To become the service account, ``sudo su account`` usually works in most environments.

.. note::

  * You may need to create the ``.ssh`` folder if it doesn't already exist.
    Be sure to set the `correct filesystem permissions for .ssh folders <https://itishermann.hashnode.dev/correct-file-permission-for-ssh-keys-and-folders>`_.

  * RSA keys are known to work well with ``nextPYP``'s `SSH client <https://github.com/mwiede/jsch>`_.
    If your organization prefers the newer ECDSA key type, or the even newer Ed25519 key type,
    you can try to generate one of those instead. Our SSH client advertises support for ECDSA and Ed25519 keys,
    but we haven't tested them ourselves just yet.

Other SSH configurations than the one suggested here may work as well. If you stray from the defaults,
you may need to update the ``config.toml`` file to describe your SSH configuration to the website.
You can find more information about all of the SSH settings in the
:doc:`full documentation for the configuration file <../reference/config>`.


Step 4: Test the new configuration
----------------------------------

After the website is restarted, go to the administration page. You can access the administration page by
clicking on your username in the upper right corner and clicking the administration link there. Or you can
just visit the administration page directly by changing the path (and hash) parts of the URL to ``/#/admin``.

On the administration page, in the *PYP* tab, click the :bdg-primary:`PYP/WebRPC Ping` button.

This button will launch a short simple job on the cluster and wait for the result.

If a pong response is returned, then the new configuration was successful.

If instead, you see an error or a timeout or a no-response message of some kind, then the configuration was not successful.
To find out what went wrong will require some debugging.

The first useful place to look for error information will be the ``micromon`` log in the ``local/logs`` folder of
your installation. Errors with the SSH connection will appear there. See :doc:`troubleshooting<./troubleshooting>` for more details.

The next place to look for errors is the log files in the ``shared/log`` folder in the shared filesystem.
If worker processes can't connect to the website, their log files will usually explain why. Usually problems
at this stage are caused by networking issues and mismatched configuration.

Getting help
------------

Getting ``nextPYP`` installed and working correctly can be tricky sometimes,
especially since everyone's needs are just a little different.
We've done our best to build an install process that's flexible enough to work in many different environments,
but sometimes things still might not work out perfectly.

If you have questions, need clarification on any of the installation options, or are just looking for a little
help getting through the installation, don't hesitate to reach out using one of options listed in the :doc:`Getting help<./known-issues>` section.
