
====================================
Installation: Attach a SLURM cluster
====================================

The `base installation instructions <./install-web.rst>`_ install complete working application that does
all the computation on the local machine.

If, after you've installed the base application, you want to improve performance by attaching a SLURM_ cluster,
these instructions will show you how.

.. _SLURM: https://slurm.schedmd.com/overview.html


Step 1: Prerequisites
---------------------

 * SLURM scheduler:
     ``nextPYP`` uses a SLURM_ compute cluster to do the data processing. The login node of the SLURM
     cluster must be reachable on the network from the machine where ``nextPYP`` is installed.

 * Shared filesystem:
     ``nextPYP`` requires that the web server and the SLURM cluster share a single filesystem (e.g.
     an NFS storage system) and it be mounted at the same mount point on every machine.
     For example, if the shared filesystem is mounted on the SLURM cluster nodes as ``/nfs/data``,
     then those files should also be available on the web server machine as ``/nfs/data``.

 * Service account:
     ``nextPYP`` requires a service account to both run the web server process, and access files on
     the shared filesystem. This user should be the same on the web server machine and the SLURM cluster.
     Because this user account runs the web server on the web server machine (which may be exposed to
     the public internet), the service account should not have administrative privileges.

 * Passwordless SSH access to the SLURM login node:
     The service account needs to have login access from the web server to the SLURM node via SSH without a password.
     This will require installing the public SSH key for the service account into the login system for the SLURM node.
     For a stock linux installation of ``sshd``, that usually means copying the public key into a file like
     `/home/account/.ssh/authorized_keys`. But for SLURM clusters with a networked login system or SSO,
     you'll need to consult your organization's IT staff for SSH key installation instructions.


Step 2: Move the shared folder to a shared filesystem
-----------------------------------------------------

By default, the installer for the application creates the shared folder inside the installation folder,
typically at ``/opt/nextPYP/shared``. This location will usually not be reachable by nodes in the SLURM cluster.

Attaching a SLURM cluster requires the shared folder be located on a filesystem that is shared between
the web server and the SLURM cluster nodes, like NFS. And the shared folder should be mounted in the same filesystem
location on each machine.

If you've mounted the shared filesystem at ``/nfs`` on each machine, create a subfolder for the shared folder, e.g.:

.. code-block:: bash

    mkdir -p /nfs/nextpyp/shared

Then move the contents of the existing shared folder to the new location.
Finally, make sure this folder and all its sub-files and folders
are readable and writable by the ``nextPYP`` service account.


Step 3: Configuration
---------------------

In the ``config.toml`` file in your installation folder, add (or change) the ``web.shared`` setting
to point to the new shared folder you just created, e.g.:

.. code-block:: toml

    [web]
    shared = '/nfs/nextpyp/shared'

You'll also need to add (or change) the ``web.host`` and/or ``web.webhost`` settings to match your network
configuration. Depending on how your network is configured, choose one of the following options.

Option 1: The SLURM cluster and the web server are on a shared private network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the application web server only binds to the loopback network interface
to prevent access from external networks. To let SLURM compute nodes access the website,
you must configure the web server to bind to the network interface used by the SLURM cluster
by configuring the ``web.host`` setting. The correct value to use here will be the hostname or IP address
that a SLURM compute node can use to connect to the web server. For example:

.. code-block:: toml

    [web]
    host = 'nextpyp.internal.myorganization.org'

.. note::

    If you're also using the reverse proxy server, you'll need to update the reverse proxy configuration
    to forward traffic to the same address you specified in the ``web.host`` setting.

    By default, the reverse proxy forwards traffic to the loopback interface, but since we just configured
    the web server to listen to a different network interface, the reverse proxy server won't be able to
    find the web server anymore.

    Change the reverse proxy server target by adding a second argument to the ``ExecStart``
    directive in the systemd unit file at ``lib/systemd/system/nextPYP-rprox.service``.
    The value of the argument should be the value of the ``web.host`` setting, e.g.:

    .. code-block::

        ExecStart="/usr/bin/nextpyp-startrprox" "nextpyp.myorganization.org" "nextpyp.internal.myorganization.org"

    If you're using a non-default value for ``web.port``, include that in the proxy target as well, e.g.:

    .. code-block::

        ExecStart="/usr/bin/nextpyp-startrprox" "nextpyp.myorganization.org" "nextpyp.internal.myorganization.org:8083"

    **TODO**: This option isn't actually supported yet on the Caddy-based reverse proxy container.
    And the Apache-based reverse proxy container receives the target setting in a different way,
    using the ``--target`` option.

.. warn::

    If the hostname or IP address you choose for the ``web.host`` setting is reachable from the public
    internet, these settings will lead to a less secure configuration and increase your risk of a
    security compromise! You should only use this configuration if the ``web.host`` value is only available
    within your private network, and not the public internet.

.. note::

    Also update your firewall settings to allow traffic from your SLURM nodes to the web server,
    over port 8080 by default, or the current value of your ``web.port`` setting.

Option 2: The SLURM cluster and the web server are only connected through the public internet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You might run into this situation if your web server and the SLURM cluster are on different
networks. In this situation, the SLURM nodes must connect to the website through the
`reverse proxy <./enable-remote-access.rst>`_ server.

To enable access in this environment, set the ``web.webhost`` setting to the public internet URL
of the web server, using the HTTPs protocol and no port number, e.g.:

.. code-block:: toml

    [web]
    webhost = 'https://nextpyp.myorganization.org'

Do not use the ``web.host`` setting in this environment. The default value here will be correct.

.. note::

    The ``web.host`` and the ``web.webhost`` settings are actually different from each other!
    Be sure not to get them confused. With the benefit of hindsight, the ``web.webhost`` setting
    would perhaps be better named ``web.url`` now, but we'd rather not make a breaking change there.

SLURM configuration
~~~~~~~~~~~~~~~~~~~

Then add a new ``[slurm]`` section to the config file as well.
At a minimum, we'll need to set the ``slurm.host`` property.

.. code-block:: toml

    [slurm]
    host = 'slurm-login.myorganization.org'

Feel free to add any other relevant SLURM configuration here as well. You can find more information about all of
the available settings in the `full documentation for the configuration file <../reference/config.html>`_.

Additonally, it may be helpful to set a few other commonly-needed options now, depending on your SLURM environment:

 * ``slurm.path``
     Path to the SLURM binaries on the login node.

 * ``slurm.queues``
     The names of any SLURM partitions to which users can submit ``nextPYP`` jobs.

 * ``slurm.gpuQueues``
     The names of any SLURM partitions with GPU hardware to which users can submit ``nextPYP`` jobs.

For example:

.. code-block:: toml

    [slurm]
    host = 'slurm-login.myorganization.org'
    path = '/opt/slurm/bin'
    queues = [ 'general', 'quick' ]
    gpuQueue = [ 'gpu' ]

After making changes to your configuration file, restart the application:

.. code-block:: bash

  sudo systemctl restart nextPYP


Step 4: SSH configuration
-------------------------

To process a compute job, the website will attempt to SSH into the login node of the SLURM cluster to submit jobs.
For this connection to work, the website must have access to an SSH key.

To generate a new SSH key for the service account, run the following commands as the service account.

.. code-block:: bash

    cd ~/.ssh
    ssh-keygen -t rsa -f id_rsa
    cat id_rsa.pub >> authorized_keys
    chmod go-w authorized_keys

.. tip::

    To become the service account, ``sudo su account`` usually works in most environments.

.. note::

    You may need to create the ``.ssh`` folder if it doesn't already exist.
    Be sure to set the
    `correct filesystem permissions for .ssh folders <https://itishermann.me/blog/correct-file-permission-for-ssh-keys-and-folders/>`_.

.. note::

    RSA keys are known to work well with nextPYP's `SSH client <http://www.jcraft.com/jsch/>`_,
    but if your organization prefers the newer ECDSA key type, you can try to generate one of those instead.
    The SSH client advertises support for ECDSA keys, but we havent tested them ourselves.

Other SSH configurations than the one suggested here may work as well. If you stray from the defaults,
you may need to update the ``config.toml`` file to describe your SSH configuration to the website.
You can find more information about all of the SSH settings in the
`full documentation for the configuration file <../reference/config.html>`_.


Step 5: Test the new configuration
----------------------------------

After the website is restarted, go to the administration page. You can access the administration page by
clicking on your username in the upper right corner and clcking the administration link there. Or you can
just visit the administration page directly by changing the path (and hash) parts of the URL to ``/#/admin``.

On the administration page, in the "PYP" tab, click the "PYP/WebRPC Ping" button.

This button will launch a short simple job on the cluster and wait for the result.

If a pong response is returned, then the new configuration was successful.

If instead, you see an error or a timeout or a no-response message of some kind, then the configuration was not successful.
To find out what went wrong will require some debugging.

The first useful place to look for error information will be the ``micromon`` log in the ``local/logs`` folder of
your installation. Errors with the SSH connection will appear there.

The next place to look for errors is the log files in the ``shared/log`` folder in the shared filesystem.
If worker processes can't connect to the website, their log files will usually explain why. Usually problems
at this stage are caused by networking issues and mismatched configuration.
