============
Getting help
============

The best way to get help is using ``nextPYP``'s `discussion board <https://github.com/orgs/nextpyp/discussions>`_. If your question hasn't been answered before, you can open a `new discussion <https://github.com/orgs/nextpyp/discussions/new/choose>`_. When reporting a problem, please share the version of ``nextPYP`` you are using and the *entire* log of the process that produced the error. To do so, copy the text from ``nextgPYP``'s log window and paste it into Github using ``code formatting`` (please avoid uploading screenshots).

Office hours
------------

For interactive help, you can join the ``nextPYP`` team during our office hours held every Friday 4:00 to 5:00 PM (EST). You can join the zoom meeting using the following `link <https://duke.zoom.us/j/96587317585>`__.


Discord server
--------------

You can also join the ``nextPYP`` community on Discord. The server is open to all users and developers of ``nextPYP``. You can join the server using the following `link <https://discord.gg/gM5sQPkb5x>`__.

Common problems
---------------

.. nextpyp:: Information on Jobs panel does not update
    :collapsible: open

    - **Problem**: When a cluster is attached, ``nextPYP`` streams information from SLURM through and HTTP-socket connection. If the connection is dropped, the information will not longer be updated on the web page. When this happens, the icon :fa:`plug` will appear at the top of the page.
    - **Solution**: Click the icon :fa:`plug` and select :bdg-primary:`Reconnect`. The icon should then change into :fa:`wifi`, indicating that the connection has been reestablished.

.. nextpyp:: Jobs run perpetually
    :collapsible: open

    - **Problem**: The Jobs panel shows that processes are running (spinning cogs) even after processing has finished.
    - **Solution**: Cancel the jobs using the red button :fa:`ban` in the jobs panel. If the problem persist, consult the :doc:`troubleshooting<../install/troubleshooting>` section and report any problems using the Github's `discussion board <https://github.com/orgs/nextpyp/discussions>`_.

.. nextpyp:: No logs to highlight, job run is too old
    :collapsible: open

    - **Problem**: When the menu option to ``Navigate to latest logs`` on a block is selected, the following toast message appears: `No logs to highlight, job run is too old`.
    - **Solution**: Go to the **Jobs** panel, open the `Older` branch and click on :bdg-primary:`Load older runs`. If you go back and select the ``Navigate to latest logs`` option, the correct log should be highlighted.

.. nextpyp:: Website is unresponsive, pages are slow to load or *Error reading from remote server* is displayed.
    :collapsible: open

    - **Problem**: This can be caused by the JVM running out of memory. You may see the following error message: ``502 Proxy Error. The proxy server received an invalid response from an upstream server. The proxy server could not handle the request. Reason: Error reading from remote server.``, and the server log may show the error: ``Caused by: java.lang.OutOfMemoryError: Java heap space.``
    - **Solution**: Increase the memory of the JVM to 8192 MB by adding the option ``heapMiB = 8192`` to the ``config.toml`` configuration file in the ``[web]`` section. Restart ``nextPYP`` for the changes to take effect.

.. nextpyp:: General unexpected behavior
    :collapsible: open

    - **Problem**: Website components are missing, pages don't load properly, etc.
    - **Solution**: We are not sure what could cause this, but here are a few things you can try: 1) reload the page, 2) clear your browser's cache, 3) update your browser, or 4) try using a different browser.

For problems with installation, please consult the :doc:`troubleshooting<../install/troubleshooting>` section.

.. admonition:: Still need help?

   Search ``nextPYP``'s `discussion board <https://github.com/orgs/nextpyp/discussions>`_ or post a new question there.