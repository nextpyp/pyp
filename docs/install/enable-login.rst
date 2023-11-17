
===================================
Installation: Enable Multiple Users
===================================

The `base installation instructions <./install-web>`_ install a complete working application for
a single person.

If, after you've installed the base application, you want to allow multiple people to access it,
these instructions will show you how to enable login accounts.


Step 1: Configuration
---------------------

In the ``config.toml`` file in your installation folder,
change the ``web.auth`` setting from ``none`` to ``login``.

.. code-block:: toml

    [web]
    auth = 'login'

After making changes to your configuration file, restart the application:

.. code-block:: bash

  sudo systemctl restart nextPYP


Step 2: Setup
-------------

When in login mode, the application will require some first-time setup to create
the first administrator account.

Open the application website in your browser. Then change the url in the address bar
by replacing the ``/#/dashboard`` part with ``/#/admin``. This will load the hidden administration page.

For example, if you access the website at the url ``https://nextpyp.myorganization.org``, then the URL
for the administrator page will be ``https://nextpyp.myorganization.org/#/admin``.

Once on the page, you will be greeted with a page like the following.

.. figure:: ../images/first_time_setup.webp

Fill out the form, click :badge:`Create Administrator,badge-primary`, and you'll make the first administrator account.

.. tip::

  See the :doc:`Administration<../reference/admin>` section for information on how to create and manage users and groups in ``nextPYP``.

From now on, to access the website, you'll need to login with your account name and password.
