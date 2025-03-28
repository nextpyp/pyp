#######################################
NYSBC course: nextPYP practical (day 2)
#######################################

This session demonstrates how explicitly optimizing for fast runtime and giving users flexibility in pre-processing steps can aid in achieving high-quality and high-throughput data acquisition in :fa:`nextPYP`. Starting from **raw data** obtained at the microscope, we'll develop an **automatic pipeline** that can perform all **pre-processing** tasks up to and including particle picking. We will demonstrate this workflow on the `EMPIAR-10164 <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_ dataset of HIV purified VLPs.

Dataset
=======

  * EMPIAR-10164: HIV Virus-Like Particles (purified VLPs)

Starting a Session
==================

.. nextpyp:: Starting a Session
  :collapsible: open

  * On your Dashboard, select the blue :bdg-primary:`Go to Sessions` button.

  * Click the blue :bdg-primary:`Start Tomography` button.

    * Give your session a user-readable name by typing in the **Name** box.

    * The **Parent Folder** box will be auto-populated with the storage location specified in your ``pyp_config.toml`` file.

      - For the workshop, this is the ``/nfs`` mount for ``bartesaghilab``.

    * Pick a *unique* **Folder Name** for your session. There can only be one folder name per session, regardless of the user-readable name!

    * Select the ``Workshop`` group.

  * Click on the :bdg-secondary:`Raw Data` tab.

    * Set **Path to raw data**: ``/nfs/bartesaghilab/nextpyp/workshop/10164/TS_*.tif``
  
  * Click on the :bdg-secondary:`Microscope Parameters` tab.

    * Set the microscope parameters as follows:

      - **Pixel size (A)**: 1.35

      - **Acceleration voltage (kV)**: 300

      - **Tilt-axis angle (degrees)**: 85.3
  
  * Click on the :bdg-secondary:`Session Settings` tab.

    * Set the session settings as follows:

      - **Number of tilts**: 41

      - **Raw data transfer**: ``link``

        - ``Link``: Create a symlink between the data on the microscope and your local computer. The data still *only* exists at the microscope.
        
        - ``Move``: Transfer the data from the microscope to your local computer, removing the data at the microscope. The data will now *only* exist on your local computer.
        
        - ``Copy``: Make a copy of the data in the microscope, and transfer the copy to your local computer. The data will now exist at both the microscope *and* your local computer.

  * Click on the :bdg-secondary:`CTF Determination` tab.

    * Set the CTF determination parameters as follows:

      - **Max resolution**: 5
  
  * Click on the :bdg-secondary:`Virion Detection` tab.

    * Set the virion detection parameters as follows:

      - **Virion radius**: 500

      - **Virion detection method**: ``auto``

      - **Spike detection method**: ``uniform``

      - **Size of equatorial band to restrict spike picking**: 800
  
  * Click on the :bdg-secondary:`Particle Detection` tab.
  
      * Set the particle detection parameters as follows:
  
        - **Detection method**: ``none``
  
        - **Detection radius**: 50

  * Click on the :bdg-secondary:`Resources` tab.
  
      * Set the resources as follows:
  
        - **Threads per task**: 41

        - **Memory per task**: 164
      
      * General advice for setting resource limits:
        
          - The **Threads per task** should match the number of tilts in your tilt series, if you have the computational resources to do so.

          - In general, the more threads you use, the more tilts that can be processed at the same time, and the faster you see pre-processing results.

          - The **Memory per task** should be set to 4 GB per thread.

  * Click :bdg-primary:`Save`, which will automatically take you to the :bdg-primary:`Operations` page.

More Features
=============

Making Changes to Pre-Processing Parameters
------------------------

.. nextpyp:: Using the ``Restart`` Option
  :collapsible: open

  *  :bdg-primary:`Restart` is a "smart" method of rerunning only what is necessary after changing pre-processing parameters.

  * Workflow: Change a parameter → :bdg-primary:`Save` settings changes → :bdg-primary:`Restart` pre-processing daemon.

  * Example: Changing the minimum distance between spikes

    * Virion detection

      - Increase **Minimum distance between spikes (voxels)** to 50

      - Click :bdg-primary:`Save`

    * Navigate to :bdg-primary:`Operations` tab

    * Click :bdg-primary:`Restart` on pre-processing daemon

    * Open :bdg-primary:`Logs` to check that the restart flag has been detected and new pre-processing jobs will be launched in response to this change.

    * Check the :bdg-secondary:`Tilt Series` tab to see that fewer particles have been picked.

.. nextpyp:: Using the ``Clear`` Option
  :collapsible: open

  * :bdg-primary:`Clear` will start pre-processing procedure from scratch.

  * This is helpful if the changes you've made touch multiple parts of the pre-processing pipeline.

    - Like re-calculating CTF or re-doing frame alignment.

Navigating the Sessions Homepage
--------------------------------

.. nextpyp:: Now to Move and Delete Sessions
  :collapsible: open

  * Sessions can be **copied** or **deleted**.

    - **CAUTION**: Deleting a session whose mode of file transfer was ``Move`` will **delete the data**.

  * Click the arrow to find where the session's network file storage location.

Importing and Exporting Sessions
--------------------------------

.. nextpyp:: Exporting a Session to a ``.star`` File
  :collapsible: open

  Sessions can be exported to ``.star`` files for downstream processing and refinement in other software (like RELION).

  * Navigate to the :bdg-Secondary:`Table` tab.

  * In the **Filters** box, type a name for your exported session.

  * Click :bdg-primary:`Export` to download the ``.star`` file.

.. nextpyp:: Importing a Session into a Project
  :collapsible: open

  Since Sessions also perform pre-processing, we can import a finished Sessions job into a project to kick-start the process of structure determination.

  * Click the :bdg-secondary:`Dashboard` link to go back to nextPYP's homepage.

  * Click the :bdg-primary:`Create New Project` button and give your project a name.

  * Click the :bdg-primary:`Import Data` button, and select the option :bdg-primary:`Tomography (from Session)`.

  * Search for the name of the session you wish to import.

  * Click the :bdg-primary:`Save` button, and then launch the job.