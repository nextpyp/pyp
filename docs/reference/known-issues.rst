============
Known Issues
============

#. General unexpected behavior

    - **Problem**: Website components are missing, pages don't load properly, etc.
    - **Solution**: Try reloading the page and most times this will correct the issue.

#. **Bad Request** error or website doesn't load

    - **Problem**: Bad Request: Your browser sent a request that this server could not understand. Size of a request header field exceeds server limit.
    - **Solution**: This can be caused by a browser incompatibility issue with Google Chrome and Safari. Try clearing up the cache or using the Mozilla Firefox browser instead.

#. Information on jobs panel does not update

    - **Problem**: ``nextPYP`` streams information from SLURM through and HTTP-socket connection. If the connection is dropped, the information will not longer be updated on the web page. When this happens, a :fa:`plug` icon will appear at the top of the page.
    - **Solution**: Click the icon :fa:`plug` and select :badge:`Reconnect,badge-primary`. The icon should then change into :fa:`wifi`, indicating that the connection has been reestablished.

#. Jobs run perpetually

    - **Problem**: The jobs panel shows that processes are running (spinning cogs) even after processing has finished.
    - **Solution**: Cancel the jobs using the red button :fa:`ban,text-danger` in the jobs panel. A page reload may also be needed.

#. Default format used by singularity containers and out-of-memory kills

    - **Problem**: There is a shortcoming in the design of squashfs, the default format used by singularity containers, which causes problems when out of memory kills target processes within a container. Developers of singularity are aware of this.
    - **Solution**: This can be prevented by making sure that there are no out-of-memory kills in your jobs by assigning more resources in the ``Job submission`` parameters. Containers can also be transformed into the ext3 format which is still supported by singularity.

#. No logs to highlight, job run is too old

    - **Problem**: When the menu option to ``Navigate to latest logs`` on a block is selected, the toast message `No logs to highlight, job run is too old` appears.
    - **Solution**: Go to the **Jobs** panel, open the `Older` branch and click on :badge:`Load older runs,badge-primary`. Repeat the **Navigate to latest logs** operation.

#. ERROR Service - Service error

    - **Problem**: The server log may shows the following error message: ``java.lang.NoClassDefFoundError: Could not initialize class java.awt.GraphicsEnvironment$LocalGE``. This error may appear if you have no graphics drivers installed in the VM running nextPYP.
    - **Solution**: Start the Java runtime into headless mode by adding the option ``-Djava.awt.headless=true`` to the website start script ``config/micromon.sh``.

#. Website is unresponsive, pages are slow to load or **Error reading from remote server** is displayed.

    - **Problem**: This can be caused by the JVM running out of memory. You may see the following error message: ``502 Proxy Error. The proxy server received an invalid response from an upstream server. The proxy server could not handle the request. Reason: Error reading from remote server.`` and the server log may show the error: ``Caused by: java.lang.OutOfMemoryError: Java heap space``.
    - **Solution**: Increase the memory of the JVM to 8192 MB by adding the option ``heapMiB = 8192`` to the ``config.toml`` configuration file in the **[web]** section. Restart nextPYP for the changes to take effect.

.. admonition:: Still need help?

   Search ``nextPYP``'s `discussion board <https://github.com/orgs/nextpyp/discussions>`_ or post a new question.