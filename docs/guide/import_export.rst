###########################
Import/export functionality
###########################

``nextPYP`` supports importing and exporting single-particle and tomography **Sessions** or **Projects**, as well as `Relion 4.0 <https://relion.readthedocs.io/en/release-4.0/>`_ or `Relion 5.0 <https://relion.readthedocs.io/en/release-5.0/>`_ projects.

Import Sessions
===============

.. nextpyp:: Importing a ``nextPYP`` session
  :collapsible: open  

  * Create a new project in ``nextPYP`` or navigate to an existing one

  .. md-tab-set::

      .. md-tab-item:: Single-particle

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Single-particle (from Session)`

        * Start typing the name of a single-particle session in the **Session** field
        
        * Select the name of session you want to import from the list of available sessions
        
        * Click :bdg-primary:`Save` and a new :bdg-secondary:`Single Particle (from Session)` block will appear on the project page

      .. md-tab-item:: Tomography

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Session)`

        * Start typing the name of a tomography session in the **Session** field
        
        * Select the name of the session you want to import from the list of available sessions
        
        * Click :bdg-primary:`Save` and a new :bdg-secondary:`Tomography (from Session)` block will appear on the project page

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

  * Once the run finishes, click inside the block to inspect the results


Import Projects
===============

.. nextpyp:: Importing a ``nextPYP`` project
  :collapsible: open  

  * Create a new project in ``nextPYP`` or navigate to an existing one

  .. md-tab-set::

      .. md-tab-item:: Single-particle

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Single-particle (from Project)`. A new :bdg-secondary:`Single Particle (from Project)` block will appear on the page

        * Select the ``Path to existing CLI project`` by clicking on the icon :fa:`search` and navigating to the location of the project you want to import
        
        * Click :bdg-primary:`Save`

      .. md-tab-item:: Tomography

        * Click on :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Project)`. A new :bdg-secondary:`Tomography (from Project)` block will appear on the page

        * Select the ``Path to existing CLI project`` by clicking on the icon :fa:`search` and navigating to the location of the tomography project you want to import
        
        * Click :bdg-primary:`Save`

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block` to launch the import process

  * Once the run finishes, click inside the block to inspect the results


Import (from \*.star)
=====================

Single-particle
---------------

.. nextpyp:: Importing a single-particle project from \*.star files
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

.. nextpyp:: Importing a tomography project from \*.star files
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


Export (to \*.star)
===================

.. nextpyp:: Exporting refinements to \*.star files
  :collapsible: open

  * Go to an existing refinement block, click on the menu icon :fa:`bars`, and select the :fa:`edit` Edit option

  * Check ``Show advanced options``

  * Go to the **Reconstruction** tab:

    .. md-tab-set::

      .. md-tab-item:: Reconstruction
  
        - Check ``Export metadata (*.star)``

  * Click :bdg-primary:`Save`

  * Click :bdg-primary:`Run` followed by :bdg-primary:`Start Run for 1 block`

  .. note::

    This will perform a single round of refinement and export the resulting metadata to a .star file. To prevent additional refinement from being executed, ensure that all refinement options are unchecked in the **Refinement** tab before running the block

    If you wish to export results from the **Reference-based refinement**, **Ab-initio reconstruction**, or **Calculate reconstruction** blocks, you have two options:

    - Re-run the original block after enabling the ``Export metadata (*.star)`` option

    - Alternatively, create and run a new **3D refinement** block downstream, making sure to check the ``Export metadata (*.star)`` option and uncheck all refinement settings in the **Refinement** tab

  Once the job ends, the results will appear in the specified folder and will be ready to import into other packages. For a tomography project, for example, you can use the `Relion's Import Coordinates <https://relion.readthedocs.io/en/release-4.0/STA_tutorial/ImportCoords.html>`_ procedure to import the data
