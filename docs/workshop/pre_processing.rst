######################
Import/export projects
######################

This tutorial shows how to import and export single-particle and tomography projects. ``nextPYP`` supports importing **Sessions**, **CLI projects**, or `Relion 4.0 <https://relion.readthedocs.io/en/release-4.0/>`_ projects. We will demonstrate how to import/export data to and from Relion (the other options work similarly).

1. Import single-particle projects
==================================

* Create or navigate to an existing single-particle project, click :bdg-primary:`Import Data` and select :bdg-primary:`Single-particle (from Star)`

* Go to the **Import parameters** tab:

  .. md-tab-set::

    .. md-tab-item:: Import parameters

      - Set the location of the ``Relion refinement star file`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

      - Set the location of the ``Relion motioncorr star file`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

      - Set the location of the ``Relion project path`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

      * Go to the **Raw data** tab:

    .. md-tab-item:: Raw data

      - Set the ``Location`` of the raw data by clicking on the icon :fa:`search` and browsing to the corresponding directory

      - Click on the **Microscope parameters** tab

    .. md-tab-item:: Microscope parameters

      - Set ``Pixel size (A)``

      - Set ``Acceleration voltage (kV)``

      * (optional) Set parameters in other tabs

* Click :bdg-primary:`Save` and the new block will appear on the project page

* Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

* Click inside the block to inspect the results


2. Import tomography projects
=============================

* Create or navigate to an existing tomography project, click :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Star)`

* Go to the **Import parameters** tab:

  .. md-tab-set::

    .. md-tab-item:: Import parameters

      - Set the location of the ``Relion refinement star file`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

      - Set the location of the ``Relion tomogram star file`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

      - Set the location of the ``Relion project path`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

      * Go to the **Raw data** tab:

    .. md-tab-item:: Raw data

      - Set the ``Location`` of the raw data by clicking on the icon :fa:`search` and browsing to the corresponding directory

      - Click on the **Microscope parameters** tab

    .. md-tab-item:: Microscope parameters

      - Set ``Pixel size (A)``

      - Set ``Acceleration voltage (kV)``

      - Set ``Tilt-axis angle (degrees)``

      * (optional) Set parameters in other tabs

* Click :bdg-primary:`Save` and the new block will appear on the project page

* Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

* Click inside the block to inspect the results


3. Export single-particle/tomography projects
=============================================

* Choose an existing :bdg-primary:`Particle refinement` block, click on the menu icon :fa:`bars, text-primary` and select the :fa:`edit, text-primary` Edit option

* Go to the **Export** tab:

  - Check ``Export metadata``

  - Set the location of the ``Input parfile`` you want to export by clicking on the icon :fa:`search` and browsing to the corresponding directory

* Click :bdg-primary:`Save`

* Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the export process

Once the job ends, the results will appear in the specified folder and will be ready to import into other packages. For a tomography project example, you can use `Relion's Import Coordinates <https://relion.readthedocs.io/en/release-4.0/STA_tutorial/ImportCoords.html>`_ procedure

.. seealso::

    * :doc:`CLI single-particle import/export<../cli/spa_import_export>`
    * :doc:`CLI tomography import/export<../cli/tomo_import_export>`
