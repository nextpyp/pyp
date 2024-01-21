============
Known Issues
============

#. **Bad Request** error or website doesn't load

    - **Problem**: Bad Request: Your browser sent a request that this server could not understand. Size of a request header field exceeds server limit.
    - **Solution**: This can be caused by a browser incompatibility issue with Google Chrome and Safari. Try clearing up the cache or using the Mozilla Firefox browser instead.

#. Information on jobs panel does not update

    - **Problem**: ``nextPYP`` streams information from SLURM through and HTTP-socket connection. If the connection is dropped, the information will not longer be updated on the web page. When this happens, the icon :fa:`plug` will appear at the top of the page.
    - **Solution**: Click the icon :fa:`plug` and select :badge:`Reconnect,badge-primary`. The icon should then change into :fa:`wifi`, indicating that the connection has been reestablished.

#. Jobs run perpetually

    - **Problem**: The jobs panel shows that processes are running (spinning cogs) even after processing has finished.
    - **Solution**: Cancel the jobs using the red button :fa:`ban,text-danger` in the jobs panel. If the problem persist, consult the :doc:`troubleshooting<../install/troubleshooting>` section and report any problems using the Github's `discussion board <https://github.com/orgs/nextpyp/discussions>`_.

#. Default format used by singularity containers and out-of-memory kills

    - **Problem**: There is a shortcoming in the design of squashfs, the default format used by singularity containers, which causes problems when out of memory kills target processes within a container. Developers of singularity are aware of this.
    - **Solution**: This can be prevented by making sure that there are no out-of-memory kills in your jobs by assigning more resources in the ``Job submission`` parameters. Containers can also be transformed into the ext3 format which is still supported by singularity.

#. No logs to highlight, job run is too old

    - **Problem**: When the menu option to ``Navigate to latest logs`` on a block is selected, the following toast message appears: `No logs to highlight, job run is too old`.
    - **Solution**: Go to the **Jobs** panel, open the `Older` branch and click on :badge:`Load older runs,badge-primary`. If you go back and select the ``Navigate to latest logs`` option, the correct log should be highlighted.

#. Website is unresponsive, pages are slow to load or **Error reading from remote server** is displayed.

    - **Problem**: This can be caused by the JVM running out of memory. You may see the following error message:

        ```
        502 Proxy Error. The proxy server received an invalid response from an upstream server.
        The proxy server could not handle the request. Reason: Error reading from remote server.
        ```

        and the server log may show the error:

        ```
        Caused by: java.lang.OutOfMemoryError: Java heap space.
        ```

    - **Solution**: Increase the memory of the JVM to 8192 MB by adding the option ``heapMiB = 8192`` to the ``config.toml`` configuration file in the ``[web]`` section. Restart ``nextPYP`` for the changes to take effect.

#. General unexpected behavior

    - **Problem**: Website components are missing, pages don't load properly, etc.
    - **Solution**: This is usually indicative of an underlying problem. Reloading the page may correct the issue temporarily, but if the problem persist, consult the :doc:`../install/troubleshooting<./troubleshooting>` section and report the problem using Github's `discussion board <https://github.com/orgs/nextpyp/discussions>`_.

For other problems, please consult the :doc:`troubleshooting<../install/troubleshooting>` section.

.. admonition:: Still need help?

   You can search ``nextPYP``'s `discussion board <https://github.com/orgs/nextpyp/discussions>`_ or post a new question.

.. important::

   When reporting a new problem, please share the *entire* log of the process that produced the error. To do so, you can either upload the file as an attachment or copy and paste the text of the log into the Github message using code formatting, for example:

.. code-block:: bash

    This is the error message I'm getting:
    ```
    10	2024-01-19 19:21:46 [INFO] Reading and converting coordinates took: 00h 00m 00s
    11	2024-01-19 19:21:48 [ERROR] An error has occurred.
    ```