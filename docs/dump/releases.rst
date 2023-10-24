
========
Releases
========

This document describes how nextPYP developers in the Bartesaghi Lab should prepare new releases.


.. note::

  This is a very WIP document and is far from complete.


Versioning
==========

Increment the version number for nextPYP in the ``/nextpyp.toml`` file.

We'll refer to this new version number through the rest of this document as ``$VERSION``.

The commit the change to the pyp repo. Then add a tag to the commit with the form ``v$VERSION``.


Building
========

Build the release distribution files using the Micromon build scripts.

TODO: be more detailed here, like the build commands, where to copy files, etc.


Documentation
=============

Build the documentation for nextPYP either manually, or by using the GitLab CI system.
When the documentation build is finished, copy the contents of the output ``html`` directory
to ``/nfs/bartesaghilab/www/files/pyp/$VERSION/docs``.

Add ``$VERSION`` to the array in ``/nfs/bartesaghilab/www/files/version-info.js``.

Create a symlink so the new documentation can see the ``versions-info.js`` file:

.. code-block:: bash

  cd /nfs/bartesaghilab/www/files/pyp
  ln -s versions-info.js $VERSION/docs/_static/

.. note::

  Be sure to replace ``$VERSION`` in the above commands with your actual version.
  Or just define the ``$VERSION`` environment variable in your shell session. That would work too.
