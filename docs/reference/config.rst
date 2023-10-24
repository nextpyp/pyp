=============
Configuration
=============

``nextPYP`` uses a single configuration file to provide all environmental information needed to run.

The file is written in TOML_.

.. _TOML: https://toml.io/en/


Placement
=========

The ``config.toml`` file can be placed anywhere you like on the filesystem, but you'll
need to let PYP know where to find it. There are two mechanisms for doing this.

1. Environment variable
-----------------------

PYP first uses the ``$PYP_CONFIG`` environment variable to find the config file.
If ``$PYP_CONFIG`` is present and its value is a path to an existing file in the filesystem,
``nextPYP`` will use that file for the configuration.

2. Default location
-------------------

Otherwise, PYP will look for the configuration file at the default location
in the user's home directory:
``~/.pyp/config.toml``


Complete Example
================

Here's an example of a minimal configuration to run ``nextPYP`` on the command line:

::

	[pyp]
	container = '/storage/singularity-images/pyp.sif'
	scratch = '/tmp/pyp-scratch/`
	binds = ['/data']

	[slurm]
	user = 'pyp'
	host = 'slurm-login-01.example.org'

Settings
========

PYP Section
-----------

``[pyp]``
~~~~~~~~~

:Required: yes
:Occurs: once

This section is for basic PYP configuration.
PYP executes inside of a Singularity_ container, so much of this configuration
is dedicated to configuring the container environment.

.. _Singularity: https://sylabs.io/guides/3.5/user-guide/introduction.html

|

``container``
~~~~~~~~~~~~~

:Type: string
:Required: yes
:Description:
	Path to the PYP Singularity container on the SLURM cluster,
	ideally in a read-only location.
:Examples:
	``container = '/storage/singularity-images/pyp.sif'``

|

``sources``
~~~~~~~~~~~

:Type: string
:Required: no
:Default: none
:Description:
	Path to a local installation of PYP sources that will be bound over the
	PYP sources inside the Singularity container.

	This is mostly only useful for developers of PYP looking for an easy way
	to edit and run PYP code.  Most users will not need this option.
:Examples:
	``sources = '/home/myuser/code/pyp'``

|

``scratch``
~~~~~~~~~~~

:Type: string
:Required: yes
:Description:
	Directory for large (multi-GiB) temporary files on the compute nodes.
	This location should have fast read/write speeds, ideally in local storage.
:Examples:
	``scratch = '/tmp/pyp-scratch``

|

``binds``
~~~~~~~~~

:Type: array of strings
:Required: no
:Default: empty list
:Description:
	List of filesystem paths to bind into the Singularity container.

	Since the PYP runs inside of a Singularity container, by default, no files
	from outside of the container will be visibile to PYP.
	To make files visible to PYP, bind the directories containing those files
	into the container.

	The following directories are not allowed to be bound because they will
	interfere with the PYP code inside the container:

	- ``/``
	- ``/bin/**``
	- ``/dev/**``
	- ``/environment/**``
	- ``/etc/**``
	- ``/lib/**``
	- ``/lib64/**``
	- ``/opt/**``
	- ``/proc/**``
	- ``/root/**``
	- ``/run/**``
	- ``/sbin/**``
	- ``/scif/**``
	- ``/singularity/**``
	- ``/srv/**``
	- ``/sys/**``
	- ``/usr/**``
	- ``/var/**``

	The ``**`` means any subdirectory under the listed directory is also not
	allowed to be bound.
:Examples:
	``binds = ['/data']``

	``binds = ['/storage1/cryoem-data', '/storage2/cryoem-data']``

|

``containerExec``
~~~~~~~~~~~

:Type: string or table
:Required: no
:Default: ``singularity``
:Description:
	The name or path of the container executable.
	To load a module, use a table with the key ``module`` instead.
	In module mode, the executable name will be the same as the module name by default.
	Override the default behavior by adding an ``exec`` key to the table with the name or path of the executable.
:Examples:
	``containerExec = '/bin/singularity``

	``containerExec = 'apptainer'``

	``containerExec = { module = 'singularity' }``

	``containerExec = { module = 'singularity', exec = 'apptainer' }``


SLURM Section
-------------

``[slurm]``
~~~~~~~~~~~

:Required: no
:Occurs: once

This section is used to configure properties of the SLURM_ cluster.

.. _SLURM: https://slurm.schedmd.com/overview.html

|

``user``
~~~~~~~~

:Type: string
:Required: yes
:Description:
	The user name PYP will use to SSH into the SLURM login node.
:Examples:
	``user = 'pyp'``

|

``host``
~~~~~~~~

:Type: string
:Required: yes
:Description:
	Hostname of a login node for the SLURM cluster.
:Examples:
	``host = 'slurm-login-01.example.org'``

|

``key``
~~~~~~~~

:Type: string
:Required: no
:Default: ``~/.ssh/id_rsa``
:Description:
	Path to SSH private key to log into the SLURM login node.
:Examples:
	``key = '/path/to/ssh/mykey'``

|

``port``
~~~~~~~~

:Type: int
:Required: no
:Default: ``22``
:Description:
	Network port to use to connect to the SSH daemon on the SLURM login node.
:Examples:
	``port = 2204``

|

``maxConnections``
~~~~~~~~~~~~~~~~~~

:Type: int
:Required: no
:Default: ``8``
:Description:
	The maximum number of simuntaneous connections to use to SSH into the SLURM login node.
	Many SSH daemons allow up to 10 connetions by default.
	Using more connections that that may require special configuration of the SSH daemon.
:Examples:
	``maxConnections = 10``

|

``timeoutSeconds``
~~~~~~~~~~~~~~~~~~

:Type: int
:Required: no
:Default: ``300``
:Description:
	The number of seconds to wait before closing an idle SSH connection.
:Examples:
	``timeoutSeconds = 500``

|

``path``
~~~~~~~~

:Type: string
:Required: no
:Default: ``/usr/bin``
:Description:
	Path to the slurm binaries on the cluster nodes.
:Examples:
	``path = '/opt/slurm/bin'``

|

``queues``
~~~~~~~~~

:Type: list of strings
:Required: no
:Default: just the default queue for the SLURM cluster
:Description:
	The names of any SLURM queues to which users can submit PYP jobs.
	If no choice is made by the user, the first queue will be used by default.
:Examples:
	``queues = ['default', 'quick']``

|

``gpuQueues``
~~~~~~~~~~~~~

:Type: list of strings
:Required: no
:Default: just the default queue for the SLURM cluster
:Description:
	The names of any SLURM queues with GPU hardware to which users can submit PYP jobs.
	If no choice is made by the user, the first queue will be used by default.
:Examples:
	``gpuQueues = ['default-gpu', 'quick-gpu']``

|

Standalone Section
------------------

``[standalone]``
~~~~~~~~~~~~~~~~

:Required: no
:Occurs: once

This section is used to configure properties of the job launcher in non-cluster (aka standalone) mode.

|

``availableCpus``
~~~~~~~~

:Type: int
:Required: no
:Default: One less than the number of processors in the system
:Description:
	The number of CPUs the standalone job launcher will use for jobs.
	You may want to set this to something less than the maximum your system supports,
	so there are always some CPU resources reserved to run the website, reverse proxy, and database processes.
:Examples:
	``availableCpus = 4``


StreamPYP Section
-----------------

``[stream]``
~~~~~~~~~~~~

:Required: yes
:Occurs: once

This section is used to configure the microscope streaming capabilities of PYP.

**TODO:** write this section


Web Section
-----------

This section is used to configure the web interface to PYP.
It is not required at all for the command line interface.

Throughout the web interface configuration, we will assume that the
server running the web interface (the *web server*) can see the same filesystem
as the SLURM nodes. Meaning, that if a file ``/data/project/file.dat`` is
visible on a SLURM node, that same file will also be visible at
``/data/project/file.dat`` on the web server.

If this is not generally true, e.g. due to networked filesystems being mounted
in different directories on different servers, there are two ways to fix it:

- Reconfigure the web server so the networked filesystems are mounted
  in the same locations as on the SLURM nodes.

- Add symbolic links to the web server filesystem so the networked filesystems
  appear to be mounted in the same locations as on the SLURM nodes.

----

``[web]``
~~~~~~~~~

:Required:
	for command line interface: no

	for web interface: yes
:Occurs: once

Minimal example:
::

	[web]
	localDir = '/home/streamPYP/web'
	sharedDir = '/network/streamPYP/shared'

|

``localDir``
~~~~~~~~~~~~

:Type: string
:Required: yes
:Description:
	Directory for the database and web server assets.

	This location should have fast read/write speeds, ideally in local storage.

	This location does not need to be sharerd with the SLURM nodes.
:Examples:
	``localDir = '/home/streamPYP/web'``

|

``sharedDir``
~~~~~~~~~~~~

:Type: string
:Required: yes
:Description:
	Directory for intermediate files and metadata for each user of the web interface.

	This location should have a lot of available space and must be also available
	on the SLURM nodes.
:Examples:
	``sharedDir = '/network/streamPYP/shared'``

|

``auth``
~~~~~~~~

:Type: string
:Required: no
:Default: ``none``
:Description:
	Which type of user authentication is used for the web interface:

	- ``login``: Users log into the web interface with a username and password.

	  Users are assigned fine-grained permissions by privileged administrator accounts.

	  This option is suitable for most users of streamPYP.

	- ``none``: No user authenticaion is performed by the web interface.

	  All visitors to the website are associated with the administrator account
	  and are granted full permissions.

	  This option is suitable for single-user instances of streamPYP or developers.

	- ``reverse-proxy``: Users are authenticated by a reverse-proxy server
	  (perhaps implementing SSO for an organization) before reaching streamPYP.

	  StreamPYP will use the ``X-userid`` HTTP header to identify users, which must
	  be securely provided by the reverse proxy server.

	  Users are assigned fine-grained permissions by privileged administrator accounts.
:Examples:
	``auth = 'login'``

|

``webhost``
~~~~~~~~~~~

:Type: string
:Required: no
:Default: ``https://$hostname``, where ``$hostname`` is the result of the ``hostname`` command
:Description:
	The URL of the webserver as seen from the SLURM nodes.

	This value should include the full URL prefix for the web server,
	including the protocol (HTTP or HTTPs) and the port number (if non-standard).

	Do not include a trailing slash.
:Examples:
	``webhost = 'https://streampyp.example.org'``
	
	``webhost = 'http://dev.streampyp.example.org:8080'``

|

``debug``
~~~~~~~~~

:Type: boolean
:Required: no
:Default: false
:Description:
	If true, enables extra features for PYP developers.

	Most users will not need this option.

|

``heapMiB``
~~~~~~~~~

:Type: integer
:Required: no
:Default: 2048
:Description:
	Number of MiB to use for the JVM heap for the website process.

	If you find the website becoming slow and less responsive,
	try allowing the website to use more memory by increasing the maximum heap size.

|

``databaseGB``
~~~~~~~~~

:Type: float
:Required: no
:Default: 1.0
:Description:
	Number of GB to use for the database cache

|

``jmx``
~~~~~~~~~

:Type: boolean
:Required: no
:Default: false
:Description:
	True to enable remote monitoring for the JVM via JMX.

	Most users will not need this option.

|

``oomdump``
~~~~~~~~~

:Type: boolean
:Required: no
:Default: false
:Description:
	True to enable heap dumps when the JVM runs out of memory.
	Heap dumps are useful to help diagnose memory issues,
	but are not needed for normal operation.

	Most users will not need this option.

|

``workflowDirs``
~~~~~~~~~

:Type: array of strings
:Required: no
:Default: empty list
:Description:
	List of folder paths containing workflow files.

	Any files found in these folders will be loaded as workflows when the web server starts.
	Any errors with reading the workflow files will be printed to the error log

:Examples:
	``workflowDirs = ['/storage/workflows']``
