=======
Support
=======

Getting help
------------

Visit GitHub's `discussion board <https://github.com/orgs/nextpyp/discussions>`_ to post questions and follow discussions


Common problems
---------------

.. nextpyp:: Bad Request error or website doesn't load
    :collapsible: open

    - **Problem**: Bad Request: Your browser sent a request that this server could not understand. Size of a request header field exceeds server limit.
    - **Solution**: This can be caused by a browser incompatibility issue with Google Chrome and Safari. Try clearing up the cache or using the Mozilla Firefox browser instead.

.. nextpyp:: Information on jobs panel does not update
    :collapsible: open

    - **Problem**: ``nextPYP`` streams information from SLURM through and HTTP-socket connection. If the connection is dropped, the information will not longer be updated on the web page. When this happens, the icon :fa:`plug` will appear at the top of the page.
    - **Solution**: Click the icon :fa:`plug` and select :bdg-primary:`Reconnect`. The icon should then change into :fa:`wifi`, indicating that the connection has been reestablished.

.. nextpyp:: Jobs run perpetually
    :collapsible: open

    - **Problem**: The jobs panel shows that processes are running (spinning cogs) even after processing has finished.
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
    - **Solution**: This is usually indicative of an underlying problem. Reloading the page may correct the issue temporarily, but if the problem persist, consult the :doc:`troubleshooting<../install/troubleshooting>` section and report the problem using Github's `discussion board <https://github.com/orgs/nextpyp/discussions>`_.

For problems with installation, please consult the :doc:`troubleshooting<../install/troubleshooting>` section.

.. admonition:: Still need help?

   You can search ``nextPYP``'s `discussion board <https://github.com/orgs/nextpyp/discussions>`_ or post a new question. When reporting a problem, please share the *entire* log of the process that produced the error. To do so, copy the text from the log and paste it into Github using code formatting (avoid uploading screenshots!).