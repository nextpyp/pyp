==============
Administration
==============

The admin page can be accessed through the url: ``https://nextpyp.myorganization.org/#/admin``

.. nextpyp:: Use this page to create, edit, or delete `Groups` and `Users`
    :collapsible: open

    .. figure:: ../images/reference_admin.webp
        :alt: Admin dashboard

Create a new Group
------------------

.. nextpyp:: To add new group
    :collapsible: open

    - Go to the :fa:`layer-group` `Groups` tab and click :bdg-primary:`+ Add Group`

    - Specify a name for the group and click :bdg-primary:`Save`

    .. figure:: ../images/reference_admin_group.webp
        :alt: Project dashboard
        :width: 350


Create a new User
-----------------

.. nextpyp:: To add new user
    :collapsible: open

    - Go to the :fa:`users` `Users` tab and click :bdg-primary:`+ Add User`

    - Specify a `User ID` and `Display Name`

    - Set the necessary permissions (the `EditSession` attribute will allow users to launch `Sessions`)

    - Assign the user to a group(s) by checking the necessary boxes

    .. figure:: ../images/reference_admin_user.webp
        :alt: Project dashboard
        :width: 350

.. tip:: Tips

    - You can also edit or delete details for `Groups` or `Users` using the :fa:`trash` and :fa:`edit` buttons
    - If you are using ``nextPYP``'s reverse-proxy for authentication, you can also create a one-time login link for new users using the :fa:`sign-in-alt` button

.. note::

    The admin page has two additional tabs which can be useful for troubleshooting installation problems. The :fa:`microscope` `PYP` tab can be used to confirm that the website can communicate with ``PYP`` over the RPC channel, and the :fa:`user-cog` `Jobs` tab can be used to troubleshoot issues with running jobs.
