==============
Administration
==============

The admin page can be accessed through the url: ``https://nextpyp.myorganization.org/#/admin``.

Using this page, administrators can create, edit or delete `Groups` and `Users` and set permissions:

.. figure:: ../images/reference_admin.webp
    :alt: Admin dashboard

Create new Group(s)
-------------------

- Go to the :fa:`layer-group,text-primary` `Groups` tab and click :badge:`+ Add Group,badge-primary`

- Specify a name for the group and click :badge:`Save,badge-primary`

.. figure:: ../images/reference_admin_group.webp
    :alt: Project dashboard
    :width: 350


Create new User(s)
------------------

- Go to the :fa:`users,text-primary` `Users` tab and click :badge:`+ Add User,badge-primary`

- Specify a `User ID` and `Display Name`

- Set the necessary permissions (the `EditSession` attribute will allow users to launch `Sessions`)

- Assign the user to a group(s) by checking the necessary boxes

.. figure:: ../images/reference_admin_user.webp
    :alt: Project dashboard
    :width: 350

.. tip::

    - You can also edit or delete details for `Groups` or `Users` using the :fa:`trash,text-primary` and :fa:`edit,text-primary` buttons

    - If you are using ``nextPYP``'s reverse-proxy for authentication, you can also create a one-time login link for new users using the :fa:`sign-in-alt,text-primary` button

.. note::

    The admin page has two additional tabs which can be useful for troubleshooting installation problems. The :fa:`microscope,text-primary` `PYP` tab can be used to confirm that the website can communicate with ``PYP`` over the RPC channel, and the :fa:`user-cog,text-primary` `Jobs` tab can be used to troubleshoot issues with running jobs.
