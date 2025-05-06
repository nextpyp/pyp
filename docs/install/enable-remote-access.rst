
====================
Enable remote access
====================

The :doc:`base installation instructions <./install-web>` install a complete working application for people using
``nextPYP`` on their local computer in a way that is secure by default.

If, after you've installed the base application, you want to allow access from over a network,
these instructions will show you how to do so securely, for a few different network configurations.


Option 1: Access within a trusted private network only
------------------------------------------------------

Choose this option if your server is not reachable from the internet,
and you'd like to access it from the local private network.

.. warning::

    If you follow these instructions, but your server *is* reachable from the public
    internet, this configuration will leave your system in an insecure state and at higher
    risk of a security compromise! Only choose this option if you're sure the server is not
    reachable from outside of your trusted private network.

Instructions
~~~~~~~~~~~~

Edit your ``config.toml`` file (in the installation directory you created) and add the ``web.host`` option:

.. code-block:: toml

    [web]
    host = '0.0.0.0'

This configuration tells the application HTTP server to bind to all available network interfaces.

After making changes to your configuration file, restart the application:

.. md-tab-set::

  .. md-tab-item:: I'm using a regular user account

    .. code-block:: bash

        ./nextpyp stop
        ./nextpyp start


  .. md-tab-item:: I'm using an administrator account

    .. code-block:: bash

        sudo systemctl restart nextPYP

To visit the website for your installation, open http://hostname:8080 in your web browser, where
``hostname`` is the network name of your server machine. The raw IP address will work here too,
e.g. http://10.0.0.5:8080.

.. tip::

    If your operating system has an active firewall, be sure to allow traffic on port 8080.


Option 2: Access through untrusted networks, like the public internet
---------------------------------------------------------------------

Choose this option if you'd like to use your website from an untrusted network, like the public internet.

Prerequisites
~~~~~~~~~~~~~

* Domain name
    Accessing the app website from an untrusted network requires your server to have a domain name, e.g., ``myserver.myorganization.org``. This method of allowing remote access will not work with raw IP addresses.

* Administrator account
    Installing an HTTP reverse proxy server requires ``root`` access on the web server machine.


Instructions
~~~~~~~~~~~~

To allow people to access your app web site securely, we'll install the reverse proxy HTTP server
that is bundled with ``nextPYP``.

First, navigate to the folder where you installed the application, e.g. ``/opt/nextPYP``:

.. code-block:: bash

  cd /opt/nextPYP

Then inspect the installation script for the reverse proxy at ``install-rprox``.
It's fairly simple. Once you're confident that it does what you want, mark it executable
and run it with administrator privileges.
You'll need to supply your server's domain name as the ``$PYP_DOMAIN`` environment variable.

.. code-block:: bash

    sudo chmod u+x install-rprox
    sudo PYP_DOMAIN=myserver.myorganization.org ./install-rprox

.. note::

    The domain name must be resolvable from the public internet, so shortcuts like ``localhost`` won't work here.
    Raw IP addresses also won't work here. The value must be a real domain name from the public internet DNS.

The install script will download the rest of the needed software components and set them up.
Assuming fast download speeds, the installation script should finish in a few minutes.


Check installation results
~~~~~~~~~~~~~~~~~~~~~~~~~~

Among other things, the installer created a ``systemd`` deamon named ``nextPYP-rprox`` to start and stop the
reverse proxy automatically. The daemon should be running now. Check it with:

.. code-block:: bash

  sudo systemctl status nextPYP-rprox

If all went well, you should be greeted with a response similar to the following:

.. code-block::

    ● nextPYP-rprox.service - nextPYP-rprox
         Loaded: loaded (/lib/systemd/system/nextPYP-rprox.service; enabled; vendor preset: enabled)
         Active: active (running) since Thu 2023-11-16 21:44:24 UTC; 21s ago
       Main PID: 3101 (starter)
          Tasks: 22 (limit: 4558)
         Memory: 58.1M
            CPU: 221ms
         CGroup: /system.slice/nextPYP-rprox.service
                 ├─3101 "Apptainer instance: root [reverse-proxy]"
                 ├─3102 appinit "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" ""
                 └─3125 caddy run --config /var/www/reverse-proxy/Caddyfile

You should be able to visit your website at the URL https://myserver.myorganization.org, where
``myserver.myorganization.org`` is the domain name you used in ``$PYP_DOMAIN``.

If not, there are a few useful places to look for debugging information. See :doc:`troubleshooting<./troubleshooting>` for more details.

.. note::

    The correct URL when using the reverse proxy will start with ``https`` rather than ``http``
    and not include a port number suffix like ``:8080``.
    No port number should be added to the URL when using the reverse proxy to access the website.


Firewall configuration
~~~~~~~~~~~~~~~~~~~~~~

The installation script will attempt to configure ``firewalld`` to allow HTTP and HTTPs traffic
from the internet. If your operating system uses a different firewall, it will not be configured by
the installation script, and you should manually configure it to allow HTTP and HTTPs traffic.


Getting help
------------

Getting ``nextPYP`` installed and working correctly can be tricky sometimes,
especially since everyone's needs are just a little different.
We've done our best to build an install process that's flexible enough to work in many different environments,
but sometimes things still might not work out perfectly.

If you have questions, need clarification on any of the installation options, or are just looking for a little
help getting through the installation, don't hesitate to reach out on our `GitHub discussions <https://github.com/orgs/nextpyp/discussions>`_ board.
