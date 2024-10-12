=====================
Enable multiple users
=====================

The :doc:`base installation instructions <./install-web>` install a complete working application for
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

.. tabbed:: I'm using a regular user account

  .. code-block:: bash

    ./nextpyp stop
    ./nextpyp start


.. tabbed:: I'm using an administrator account

  .. code-block:: bash

    sudo systemctl restart nextPYP


Step 2: Setup
-------------

When in ``login`` mode, the application will require some first-time setup to create
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


Getting Help
------------

Getting ``nextPYP`` installed and working correctly can be tricky sometimes,
especially since everyone's needs are just a little different.
We've done our best to build an install process that's flexible enough to work in many different environments,
but sometimes things still might not work out perfectly.

If you have questions, need clarification on any of the installation options, or are just looking for a little
help getting through the installation, don't hesitate to reach out on our `GitHub discussions <https://github.com/orgs/nextpyp/discussions>`_ board.
