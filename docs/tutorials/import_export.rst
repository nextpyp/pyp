######################
Import/export projects
######################

This page shows how to import and export single-particle and tomography projects. ``nextPYP`` supports importing projects from existing nextPYP **Sessions** or **Projects**, as well as `Relion 4.0 <https://relion.readthedocs.io/en/release-4.0/>`_ or `Relion 5.0 <https://relion.readthedocs.io/en/release-5.0/>`_ projects.

Import nextPYP sessions
========================

.. nextpyp:: Import a session
  :collapsible: open  

  * Create a new project in ``nextPYP`` or navigate to an existing one

  .. md-tab-set::

      .. md-tab-item:: Single-particle

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Single-particle (from Sessions)`

        * Start typing the name of a single-particle session in the **Session** field
        
        * Select the name of session you want to import from the list of available sessions
        
        * Click :bdg-primary:`Save` and a new :bdg-secondary:`Single Particle (from Sessions)` block will appear on the project page

      .. md-tab-item:: Tomography

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Sessions)`

        * Start typing the name of a tomography session in the **Session** field
        
        * Select the name of the session you want to import from the list of available sessions
        
        * Click :bdg-primary:`Save` and a new :bdg-secondary:`Tomography (from Sessions)` block will appear on the project page

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

  * Once the run finishes, click inside the block to inspect the results


Import nextPYP projects
=======================

.. nextpyp:: Import a project
  :collapsible: open  

  * Create a new project in ``nextPYP`` or navigate to an existing one

  .. md-tab-set::

      .. md-tab-item:: Single-particle

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Single-particle (from Project)`. A new :bdg-secondary:`Single Particle (from Project)` block will appear on the project page

        * Select the ``Path to existing CLI project`` by clicking on the icon :fa:`search` and navigating to the location of the project you want to import
        
        * Click :bdg-primary:`Save`

      .. md-tab-item:: Tomography

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Project)`. A new :bdg-secondary:`Tomography (from Project)` block will appear on the project page

        * Select the ``Path to existing CLI project`` by clicking on the icon :fa:`search` and navigating to the location of the tomography project you want to import
        
        * Click :bdg-primary:`Save`

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

  * Once the run finishes, click inside the block to inspect the results


Import refinements in \*.star format
====================================

Single-particle
---------------

.. nextpyp:: Import single-particle refinement from \*.star files
  :collapsible: open  

  * Create or navigate to an existing project in ``nextPYP``
  
  * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Single-particle (from Star)`

  * Go to the **Import parameters** tab:

    .. md-tab-set::

      .. md-tab-item:: Import parameters

        - Set the ``Relion project path`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

        - Set the location of the ``3D refinement file (*.star)`` by clicking on the icon :fa:`search` and browsing to the corresponding file

        - (optional) Set the location of the ``Motion file (*.star)`` by clicking on the icon :fa:`search` and browsing to the corresponding file

        * Go to the **Raw data** tab:

      .. md-tab-item:: Raw data

        - Set the ``Location`` of the raw data by clicking on the icon :fa:`search` and browsing to the corresponding directory

        - Click on the **Microscope parameters** tab

      .. md-tab-item:: Microscope parameters

        - Set ``Pixel size (A)``

        - Set ``Acceleration voltage (kV)``

        * (optional) Set parameters in other tabs as needed

  * Click :bdg-primary:`Save` and a new :bdg-secondary:`Single Particle (from star)` block will appear on the project page

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

  * Once the run finishes, click inside the :bdg-secondary:`Single Particle (from star)` block to inspect the results


Tomography
----------

.. nextpyp:: Import tomography refinement from \*.star files
  :collapsible: open  

  * Create or navigate to an existing project in ``nextPYP``
  
  * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Star)`

  * Go to the **Import parameters** tab:

    .. md-tab-set::

      .. md-tab-item:: Import parameters

        - Set the ``Relion project path`` by clicking on the icon :fa:`search` and browsing to the corresponding directory

        - Set the location of the ``Tomograms file (*.star)`` by clicking on the icon :fa:`search` and browsing to the corresponding file

        - Set the location of the ``3D refinement file (*.star)`` by clicking on the icon :fa:`search` and browsing to the corresponding file

        - Select the ``Relion version`` used to generate the star file (4.0 or 5.0)

        * Go to the **Raw data** tab:

      .. md-tab-item:: Raw data

        - Set the ``Location`` of the raw data by clicking on the icon :fa:`search` and browsing to the corresponding directory

        - Click on the **Microscope parameters** tab

      .. md-tab-item:: Microscope parameters

        - Set ``Pixel size (A)``

        - Set ``Acceleration voltage (kV)``

        - Set ``Tilt-axis angle (degrees)``

        * (optional) Set parameters in other tabs as needed

  * Click :bdg-primary:`Save` and a new :bdg-secondary:`Tomography (from star)` block will appear on the project page

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

  * Once the run finishes, click inside the :bdg-secondary:`Tomography (from star)` block to inspect the results


Export refinements in \*.star format
====================================

.. nextpyp:: Export 3D refinement in \*.star format
  :collapsible: open

  * Go to an existing refinement block, click on the menu icon :fa:`bars`, and select the :fa:`edit` Edit option

  * Check ``Show advanced options``

  * Go to the **Reconstruction** tab:

    .. md-tab-set::

      .. md-tab-item:: Reconstruction
  
        - Check ``Export metadata (*.star)``

  * Click :bdg-primary:`Save`

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block`

  This will execute one round of refinement and export the resulting metadata to a \*.star file. To avoid running any additional refinement, make sure to uncheck any refinement options in the **Refinement** tab before running the block. If you want to export results from the **Reference-based refinement**, **Ab-initio reconstruction**, or **Calculate reconstruction** blocks, you can either re-run the blocks (after selecting the option to export metadata as indicated above), or create and run new **3D refinement** block downstream (after checking the ``Export metadata (*.star)`` option and unchecking any refinement options in the **Refinement** tab)

  Once the job ends, the results will appear in the specified folder and will be ready to import into other packages. For a tomography project, for example, you can use the `Relion's Import Coordinates <https://relion.readthedocs.io/en/release-4.0/STA_tutorial/ImportCoords.html>`_ procedure to import the data
