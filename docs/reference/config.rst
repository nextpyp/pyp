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

``nextPYP`` first uses the ``$PYP_CONFIG`` environment variable to find the config file.
If ``$PYP_CONFIG`` is present and its value is a path to an existing file in the filesystem,
``nextPYP`` will use that file for the configuration.

2. Default location
-------------------

Otherwise, ``nextPYP`` will look for the configuration file at the default location
in the user's home directory:
``~/.pyp/config.toml``


Complete Example
================

Here's an example of a minimal configuration to run ``nextPYP`` on the command line:

::

	[pyp]
	container = '/storage/singularity-images/pyp.sif'
	scratch = '/tmp/pyp-scratch/'
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
	``scratch = '/tmp/pyp-scratch'``

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
~~~~~~~~~~~~~~~~~

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
:Required: no
:Default: The username of the website process
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
~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

:Type: int
:Required: no
:Default: One less than the number of processors in the system
:Description:
	The number of CPUs the standalone job launcher will use for jobs.
	You may want to set this to something less than the maximum your system supports,
	so there are always some CPU resources reserved to run the website, reverse proxy, database processes,
	and anything else your system needs to run.
:Examples:
	``availableCpus = 4``

|

``availabileMemoryGiB``
~~~~~~~~~~~~~~~~~~~~~~~

:Type: int
:Required: no
:Default: 80% of the available total memory
:Description:
	The amount of memory, in GiB, the standalone job launcher will use for jobs.
	You may want to set this to something less than the maximum your system supports,
	so there is always some leftover memory to run the website, reverse proxy, database processes,
	and anything else your system needs to run.
:Examples:
	``availabileMemoryGiB = 4``

|

``availableGpus``
~~~~~~~~~~~~~~~~~~~~~~~

:Type: int
:Required: no
:Default: The total number of NVidia GPUs in your system
:Description:
	The number of NVidia GPUs the standalone job launcher will use for jobs.
	AMD, Intel, and other GPU types aren't supported yet,
	unless they somehow are visible to and usable by the NVidia Cuda runtime.
:Examples:
	``availableGpus = 4``

|

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

``host``
~~~~~~~~

:Type: string
:Required: no
:Default: ``127.0.0.1``
:Description:
	The network interface to which the application HTTP server (not the reverse proxy HTTP server) should bind.

	By default, the application HTTP server binds only to the loopback network interface, i.e. localhost,
	so the application HTTP server will only be reachable from the local computer.

	To make the application HTTP server reachable from an external private network, set ``host`` to ``0.0.0.0`` to bind
	to all available network interfaces.

	.. warning::
		The application HTTP server is not designed to securely handle traffic from the public internet.
		Exposing the application HTTP server directly to the public internet increases your risk of a security
		compromise.

		Only set ``host`` to ``0.0.0.0`` if the application HTTP server is isolated from the public internet
		by a firewall or a private network.

		To make the website securely accessible from the public internet, install the reverse proxy HTTP server
		that is bundled with nextPYP, which is designed to operate securely in that environment.

:Examples:
	``host = '10.0.3.4'``

	``host = '0.0.0.0'``

|

``port``
~~~~~~~~

:Type: int
:Required: no
:Default: ``8080``
:Description:
	The network port to which the application HTTP server (not the reverse proxy HTTP server) should bind.

	By default, the application HTTP server binds to the port 8080, which is an unofficial secondary port for HTTP traffic.
	Since port 8080 is not a privileged port, the application HTTP server can run without root privileges.

	Since port 8080 is a common port for locally-running HTTP applications, you may already have another
	service installed that uses that port. To avoid a port conflict, you can configure the application HTTP
	server to use a different port, but be sure to use a non-privileged port at or above 1024.

	.. warning::
		Using a privileged port (below 1024) like 80 or 443 for the application server requires root privileges,
		but the application HTTP server was not designed to run with root privileges. Doing so would be insecure,
		and any security compromise that had access to elevated permissions would be much more severe.
		If you wish to run the website on a canonical HTTP port like 80 or 443, you should use the reverse proxy
		HTTP server bundled with nextPYP, which is designed to operate securely when exposed to the public internet.

:Examples:
	``port = 8082``

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
~~~~~~~~~~~~~

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
:Default: ``http://$host:$port``, where ``$host`` and ``$port`` are the values of the ``web.host`` and ``web.port`` configuration values respectively.
:Description:
	The URL of the webserver as visible from the pyp process.

	This value should include the full URL prefix for the web server,
	including the protocol (HTTP or HTTPs) and the port number (if non-standard).
	Do not include a trailing slash.

	When running in standalone mode, the pyp process will run on the same machine as the web server.
	In this environment, the default value will be correct, and there should be no need to choose a different value.

	When the pyp process runs on an external compute node (in, say, a SLURM cluster), this value must be the URL of
	the website from the point of view of the compute node. The default value will not be correct in this case,
	so be sure to set ``webhost`` to the correct value for your environment.

	If the compute node is on a private network that is shared with the web server, then the correct value of
	``webhost`` will be ``http://$hostname:$port`` where ``$port`` is the ``web.port`` configuration value and
	``$hostname`` is the host name of the server from the point of view of the compute node. Note this configuration
	uses unencrypted HTTP rather than encrypted HTTPs.

	.. warning::
		In this private network configuration, if the web server has any public network interfaces,
		be sure to configure the firewall to only allow connections to the port defined by ``web.port``
		over the private network interface. Connections over the public network interface should be blocked
		by the firewall.

	If the compute node is not on a shared private network with the web server, then the correct value will be
	``https://$domain`` where ``$domain`` is the domain name of the web server as configured in the DNS registry.
	This configuration requires using the reverse proxy HTTP server bundled with nextPYP to enable encrypted HTTPs
	connections, since the connection may travel over an untrusted network, like the public internet.

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
~~~~~~~~~~~

:Type: integer
:Required: no
:Default: 2048
:Description:
	Number of MiB to use for the JVM heap for the website process.

	If you find the website becoming slow and less responsive,
	try allowing the website to use more memory by increasing the maximum heap size.

|

``databaseGB``
~~~~~~~~~~~~~~

:Type: float
:Required: no
:Default: 1.0
:Description:
	Number of GB to use for the database cache

|

``jmx``
~~~~~~~

:Type: boolean
:Required: no
:Default: false
:Description:
	True to enable remote monitoring for the JVM via JMX.

	Most users will not need this option.

|

``oomdump``
~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~

:Type: array of strings
:Required: no
:Default: empty list
:Description:
	List of folder paths containing workflow files.

	Any files found in these folders will be loaded as workflows when the web server starts.
	Any errors with reading the workflow files will be printed to the error log

:Examples:
	``workflowDirs = ['/storage/workflows']``

|

``minPasswordLength``
~~~~~~~~~~~~~~~~~~~~~

:Type: integer
:Required: no
:Default: 12
:Description:
	The minimum length accepted for new passwords.
