###############################################
NYSBC Workshop: nextPYP Practical
###############################################

We will demonstrate how explicitly optimizing for fast runtime and giving users flexibility in pre-processing steps can aid in achieving high-quality and high-throughput data acquisition in :fa:`nextPYP`. Starting from **raw data** obtained at the microscope, we'll develop an **automatic pipeline** that can perform all **pre-processing** tasks up to and including particle picking. We will demonstrate this workflow on the `EMPIAR-10164 <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_ dataset of HIV purified VLPs.

Dataset
=======

  * EMPIAR-10164: HIV Virus-Like Particles (purified VLPs)

Create a Session
----------------

.. nextpyp:: 
  :collapsible: open

  * On your Dashboard, select the blue :bdg-primary:`Go to Sessions` button.

  * Click the blue :bdg-primary:`Start Tomography` button.

Session Settings
----------------

.. nextpyp:: 
  :collapsible: open

  * Give your session a user-readable name by typing in the ``Name`` box.

  * The ``Parent Folder`` box will be auto-populated with the storage location specified in your ``pyp_config.toml`` file.

    - For the workshop, this is the ``/nfs`` mount for ``bartesaghilab``.

  * Pick a *unique* ``Folder Name`` for your session. There can only be one folder name per session, regardless of the user-readable name!

  * Select the ``Workshop`` group

Raw Data
--------

.. nextpyp:: 
  :collapsible: open

  * Path to raw data: ``/nfs/bartesaghilab/nextpyp/workshop/10164/TS_*.tif``

Microscope Parameters
---------------------

.. nextpyp:: 
  :collapsible: open

  * Pixel size: 1.35

  * Acceleration voltage: 300

  * Tilt-axis angle: 85.3

Session Settings (continued)
----------------------------

.. nextpyp:: 
  :collapsible: open

  * Number of tilts: 41

  * Raw data transfer: ``link``

    - ``Link``: Create a symlink between the data on the microscope and your local computer. The data still *only* exists at the microscope.
    
    - ``Move``: Transfer the data from the microscope to your local computer, removing the data at the microscope. The data will now *only* exist on your local computer.
    
    - ``Copy``: Make a copy of the data in the microscope, and transfer the copy to your local computer. The data will now exist at both the microscope *and* your local computer.

CTF Determination
-----------------

.. nextpyp:: 
  :collapsible: open

  * Max resolution: 5

Virion Detection
----------------

.. nextpyp:: 
  :collapsible: open

  * Virion radius: 500

  * Virion detection method: ``auto``

  * Spike detection method: ``uniform``

  * Minimum distance between spikes: 8

  * Size of equatorial band to restrict spike picking: 800

Particle Detection
------------------

.. nextpyp:: 
  :collapsible: open

  * Detection method: ``none``

    - Remember that we have just picked our "particles" (virions) in the previous tab!

  * Detection radius: 50

Resources
---------

.. nextpyp:: 
  :collapsible: open

  These settings apply for all datasets:

  * Threads per task: 41

    - This number should match the number of tilts in your tilt series.

    - In general, the more threads you use, the more tilts that can be processed at the same time, and the faster you see pre-processing results.
  
  * Memory per task: 164

    - As a rule of thumb, use 4x as much memory as you have threads.

More Features
=============

Using the Restart Option
------------------------

.. nextpyp:: 
  :collapsible: open

  * "Smart" method of rerunning only what is necessary after changing pre-processing parameters

  * Workflow: Change a parameter → ``Save`` settings changes → ``Restart`` Pre-processing daemon

  .. nextpyp:: 
    :collapsible: open
    :title: Example: Changing the minimum distance between spikes

    * Virion detection

      - Increase ``Minimum distance between spikes (voxels)`` to 20

      - Click ``Save``

    * Navigate to ``Operations`` tab

    * Click ``Restart`` on pre-processing daemon

    * Open ``Logs`` to check that the restart flag has been detected and new pre-processing jobs will be launched in response to this change

    * Check ``Tilt series`` tab to see that fewer particles have been picked

Using the Clear Option
----------------------

.. nextpyp:: 
  :collapsible: open

  * Start pre-processing procedure from scratch

  * Helpful if the changes you've made touch multiple parts of the pre-processing pipeline

    - Like re-calculating CTF or re-doing frame alignment

Navigating the Sessions Homepage
--------------------------------

.. nextpyp:: 
  :collapsible: open

  * Sessions can be **copied** or **deleted**

    - **CAUTION**: Deleting a session whose mode of file transfer was ``Move`` will **delete the data**.
    
  * Click the arrow to find where the session's network file storage location
