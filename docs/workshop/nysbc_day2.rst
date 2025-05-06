#######################################
NYSBC course: nextPYP practical (day 2)
#######################################

This session demonstrates how explicitly optimizing for fast runtime and giving users flexibility in pre-processing steps can aid in achieving high-quality and high-throughput data acquisition in ```nextPYP``. 

Starting from **raw data** obtained at the microscope, we'll build an **automatic pipeline** that can perform all **pre-processing** tasks up to and including particle picking.

Dataset
=======

For this session we will use the `EMPIAR-10164 <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_ dataset of HIV-1 purified VLPs.

On-the-fly pre-processing
=========================

Creating/starting new sessions
------------------------------

.. nextpyp:: Sessions allow pre-processing of tilt-series on-the-fly
  :collapsible: open

  * On your Dashboard, select the :bdg-primary:`Go to Sessions` button.

  * Click the :bdg-primary:`+ Start Tomography` button.

    * Give your session a user-readable name by typing in the **Name** box.

    * The **Parent folder** box will be auto-populated with a default location to store the data.

    * Pick a *unique* **Folder name** for your session. There can only be one folder name per session, regardless of the user-readable name!

    * Select the ``Workshop`` group.

  * Click on the **Raw data** tab.

    * Set ``Path to raw data`` to ``/nfs/bartesaghilab/nextpyp/workshop/10164/TS_*.tif``
  
  * Click on the **Microscope parameters** tab.

    - Set ``Pixel size (A)`` to 1.35

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 85.3
  
  * Click on the **Session settings** tab.

    - Set ``Number of tilts`` to 41

    - Set ``Raw data transfer`` to link

      - ``Link``: Create a symlink between the data on the microscope and the Session folder. The data still *only* exists at the microscope.
      
      - ``Move``: Transfer the data from the microscope to the Session folder, removing the data at the microscope. The data will now *only* exist on the Sessions folder.
      
      - ``Copy``: Make a copy of the data in the microscope to your Session folder. The data will now exist at both the microscope *and* your Session folder.

  * Click on the **CTF Determination** tab.

    - Set ``Max resolution`` to 5
  
  * Click on the **Virion detection** tab.

    - Set ``Virion radius`` to 500

    - Set ``Virion detection method`` to auto

    - Set ``Spike detection method`` to uniform

    - Set ``Size of equatorial band to restrict spike picking`` to 800
  
  * Click on the **Particle detection** tab.
  
    - Set ``Detection method`` to none

    - Set ``Detection radius`` to 50

  * Click on the **Resources** tab.
  
    - Set ``Split, Threads`` to 41

    * General advice for setting resource limits:
      
      - The ``Split, Threads`` should match the number of tilts in your tilt series, if you have the computational resources to do so.

      - In general, the more threads you use, the more tilts that can be processed at the same time, and the faster you see pre-processing results.

  * Click :bdg-primary:`Save`, which will automatically take you to the :bdg-primary:`Operations` page.

  * Click :bdg-primary:`Start` to launch the session.

Restarting sessions
-------------------

.. nextpyp:: Use the ``Restart`` option to make changes to ongoing Sessions
  :collapsible: open

  *  :bdg-primary:`Restart` is a "smart" method of re-running only what is necessary after changing pre-processing parameters.

  * Workflow: Change a parameter → :bdg-primary:`Save` settings changes → :bdg-primary:`Restart` pre-processing daemon.

  * Example: Changing the minimum distance between spikes

    * Go to the **Virion detection** tab

    * Increase **Minimum distance between spikes (voxels)** to 50

    * Click :bdg-primary:`Save`

    * Navigate to :bdg-primary:`Operations` tab

    * Click :bdg-primary:`Restart` on pre-processing daemon

    * Open :bdg-primary:`Logs` to check that the restart flag has been detected and new pre-processing jobs will be launched in response to this change.

    * Check the **Tilt Series** tab to see that fewer particles have been picked.

.. nextpyp:: Using the ``Clear`` option
  :collapsible: open

  * :bdg-primary:`Clear` will start pre-processing procedure from scratch

  * This is helpful if you want to start fresh making sure any previous pre-processing results are ignored.

Copying/deleting sessions
-------------------------

.. nextpyp:: Quickly create a session by copying an existing one
  :collapsible: open

  * Sessions can be **copied** or **deleted**.

  * Click the icon :fa:`location-arrow` to find the session's file storage location.

.. warning::

  Deleting a session whose mode of file transfer was set to ``Move`` will **delete all files (including the raw data)!**.

Importing/exporting sessions
----------------------------

.. nextpyp:: Exporting a session in ``.star`` format
  :collapsible: open

  Sessions can be exported in ``.star`` format for downstream processing and refinement in other software.

  * Navigate to the :bdg-Secondary:`Table` tab.

  * In the **Filters** box, type a name for your exported session.

  * Click :bdg-primary:`Export` to launch the export job. The job's log will indicate the location of the exported ``.star`` file.

.. nextpyp:: Importing a session into a project
  :collapsible: open

  Since Sessions also perform pre-processing, we can import a finished Session into a project to kick-start the process of structure determination.

  * Click the :bdg-secondary:`Dashboard` link to go back to nextPYP's homepage.

  * Click the :bdg-primary:`Create New Project` button and give your project a name.

  * Click the :bdg-primary:`Import Data` button, and select the option :bdg-primary:`Tomography (from Session)`.

  * Search for the name of the session you wish to import.

  * Click the :bdg-primary:`Save` button, and then launch the job.

Day 2 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  In this session we learned how to run on-the-fly pre-processing in ``nextPYP``:
  
  * Continuously monitor raw data folder for incoming tilt-series

  * Raw data transfer (copy, link, move)

  * Pre-processing (frame alignment, tilt-series alignment, CTF estimation, and tomogram reconstruction)

  * Particle picking (geometry-based, size-based, etc.)

  * Restart, clear, copy or delete sessions

  * Import and export sessions

  Feel free to revisit :doc:`day 1<nysbc_day1>` or explore other options available in ``nextPYP`` as described in the :doc:`User Guide<../guide/overview>`.