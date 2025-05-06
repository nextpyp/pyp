==========================
SLURM submission templates
==========================

SLURM job submission can be customized in ``nextPYP`` by creating different submission templates for different situations.
These template files generate a batch script that gets sent to ``sbatch`` during job submission. The primary way to
configure SLURM during ``sbatch`` job submission is by emitting ``#BATCH`` directives into the shell script.

During cluster attachment, you already downloaded the default template. The default template is a very basic template
that submits jobs using only values chosen in ``nextPYP``'s job submission user interface. This basic template
may work as-is for some SLURM clusters, but if it doesn't work for your specific cluster, feel free to edit the
template to suit your needs.


Template language
-----------------

The templating system used by ``nextPYP`` is called `Pebble`_ and uses a syntax that's very similar to the popular
`Jinja`_ syntax. If you've used other templating systems before, hopefully templates in ``nextPYP`` feel very
familiar to you.

.. _Pebble: https://pebbletemplates.io/
.. _Jinja: https://jinja.palletsprojects.com/en/stable/templates/

To get started writing your own templates, consult the `official template language`_ reference from Pebble.

.. _official template language: https://pebbletemplates.io/wiki/guide/basic-usage/

Create new templates by adding ``.peb`` files to the templates folder you created during cluster attachment.
Template files are read and processed at each job launch, so changes to your template files will take effect
the next time you launch a job in ``nextPYP``. There's no need to restart the ``nextPYP`` daemon to apply template
changes.


Template front-matter
---------------------

Template files have a metadata section at the top called "front-matter" that provides information to ``nextPYP``
to help users choose which template to use when submitting a job.

The front-matter section is parsed as any lines of text between two delimiting lines at the top of the template
file that look like this:

.. code-block::

  #---

So for example:

.. code-block::

  #--
  This is front-matter
  #--
  This is the template

The text in between these two delimiting lines is a `TOML`_ document that describes the template metadata.
Currently, only a few properties from the front-matter are used by ``nextPYP``:

.. _TOML: https://toml.io/en/

* title

  A short name for the template, to be shown in user interfaces for selecting templates.

* description

  A longer description of the template, usually shown alongside the title, when space in the UI allows it.

Unrecognized TOML values (like ``comment``) will be ignored by ``nextPYP``,
so these values could be used to store metadata that is specific to your organization without causing any
errors in ``nextPYP``.

An example front-matter section, including the delimiting lines and the intervening TOML document,
might look something like this:

.. code-block:: toml

  #--
  title = "The Best template"
  description = "This template is the best one. Use it instead of that other worse template over there."
  comment = "nextPYP won't read this comment, but the humans looking at this file might."
  #--


Templates create shell scripts
------------------------------

Since the templates are used to generate shell scripts, the first line of the template must be a shell `shebang`_,
such as:

.. code-block:: bash

  #!/bin/sh

The default template uses ``#!/bin/bash``, since the Bash shell has many useful features and is widely supported
on many systems, but you could customize your templates to use whatever shell is available in your computing
environment, or just defer to the default shell configured by the operating system, at ``/bin/sh``.

.. _shebang: https://en.wikipedia.org/wiki/Shebang_(Unix)


Variables
---------

Template files can access a variety of different variables during evaluation, available in two different namespaces:
``job`` and ``user``.


Job variables
-------------

The main collection of variables
can be accessed from the ``job`` namespace. Variables here expose the SLURM options that ``nextPYP`` chose
when launching a job, often with help from the user. These variable names match the arguments to ``sbatch``.
The current list of ``sbatch`` variables used by ``nextPYP`` include:

* ``name``
* ``dependency``
* ``array``
* ``cpus-per-task``
* ``mem``
* ``time``
* ``gres``

Variable names are case-sensitive, and for any given job, not all of these variables will exist.
Using a variable that does not exist in a template will result in an error during template evaluation,
so you should guard usage of ``job`` variables by using conditional logic and the ``exists`` filter.

See the default template for extensive examples of how to access and render ``job`` variables.


User variables
--------------

The second collection of variables can be accessed from the ``user`` namespace, and reflect the user submitting
the job, rather than any properties of the job itself. The following variables are supported:

* ``os_username``
  If configured in the admin page, this variable will contain the connected operating system username of the
  ``nextPYP`` user. If not configured, or there is no user associated with the job submission,
  this variable will not exist.

* ``properties``
  ``nextPYP`` supports defining arbitrary values with user accounts and those values are accessible to the template
  system via this ``user.properties`` namespace.

  To configure a user property, edit that user in the :doc:`admin page<../reference/admin>`. At the bottom of the
  edit panel, you'll find a section called "Custom Properties".

  .. figure:: ../images/user_properties.webp
    :align: center
    :height: 300

    The edit user panel, at the bottom, showing the Custom Properties section.

  Adding key-value pairs to this section, and then clicking :bdg-primary:`Save` will
  allow you to use those values in templates. For example, if you created a property called ``nodes``
  with a value of ``intel`` for a user, then when that user submits a job, you can access that property in a
  template through the ``user.properties.nodes`` variable.

  If the job has no user associated with it, the ``user.properties`` variable will not exist, so it's often
  a good idea to guard your usages of the variable with conditional logic and the ``exists`` filter.
  It may be helpful to guard usage of individual user properties the same way, if you don't expect
  every user to have each property defined.


Job commands
------------

Every template should have the following syntax that renders the actual commands of the job into the shell script.

.. code-block::

  {{ job.commands | raw }}

This syntax should be the last line in the template file. Omitting it will cause the job launch to fail.

.. note::

  The job commands are passed through the ``raw`` filter before rendering. By default, the template engine in
  ``nextPYP`` escapes and sanitizes variable values using POSIX shell quoting rules, to prevent injection attacks
  into shell scripts from these variables. The ``raw`` filter skips the escaping step, since we expect
  this variable to contain legitimate shell commands.


Template debugging
------------------

When building your templates, hopefully everything works perfectly the first time, and your SLURM jobs launch
and run without issue. For the other times when that's not the case, you can debug the template evaluation step
by looking at the "Launch" tab of your job.

Any errors during the launch procedure will be shown at the top, in the "LAUNCH INFO" section:

.. figure:: ../images/template_failure.webp
  :align: center

  A failed job launch, with template debugging info shown as the Reason.

And if template generation is successful, the generated shell script, with all conditional logic evaluated and
all varialbes rendered, will be shown at the bottom of the tab, in the "SCRIPT" section:

.. figure:: ../images/template_success.webp
  :align: center

  A successful template evaluation, showing the first part where the ``sbatch`` directives are rendered.

