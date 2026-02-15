###################
DHVI course (day 4)
###################

Session 1: Map post-processing and visualization
================================================

Part 1 - Map post-processing
----------------------------

.. nextpyp:: Step 1: Post-processing
  :collapsible: open
  
  * Click on ``Frames`` (output of the :bdg-secondary:`Movie refinement` block) and select :bdg-primary:`Post-processing`

  * Go to the **Post-processing** tab

    - Next to ``First half map (*_half1.mrc)`` click the :fa:`search` icon. Select the ``*_half1.mrc`` file and click :bdg-primary:`Choose File`

    - Set ``Masking method`` to from file usign the dropdown menu

    - Next to ``Mask file (*.mrc)`` click the :fa:`search` icon. Browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc`` and click :bdg-primary:`Choose File`

    - Set the ``B-factor method`` to adhoc using the dropdown menu

    - Set the ``Adhoc value (A^2)`` to -25 

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step 2: Map and model assessment in ChimeraX
  :collapsible: open
  
  * I will be using a prealigned pdb file and files downloaded from nextPYP to demonstrate how one can visualize their final map aligned to a model in Chimera. 

  * Download files

    - In the :bdg-secondary:`Post-processing` block, go to the **Reconstruction** tab. Click on the drop down menu **Select an MRC file to download**. Select the Full-Size Map. Your browser will download the post processed map as an MRC file. 

    - We are using a pre-aligned, pre-cropped pdb file (5L93) so do not need to download this. For your experiments, you would download whatever model required. 
  
    - Open the downloaded MRC file in Chimera. Visualize your beautiful map. To get a better look at your map/model fitting, open an atomic model in Chimera. Under the **Map** tab, Click **Zone**. Note we are left with a slightly larger zone than we would like so we will copy the zone command from the output to the terminal line, and edit the range. This leaves us with: 

    .. code-block:: bash 

      volume zone #2 nearAtoms #1 range 2.4

    - Select the model, go to **Actions**, **Atoms/Bonds**, and **Show Sidechain/Base**
    
    - You can now view the model fit to your map interactively in ChimeraX


Part 2: Visualization of results in ArtiaX/ChimeraX
---------------------------------------------------

.. nextpyp:: Step 1: Download all the necessary files
  :collapsible: open

  - Select a tomogram you wish to visualize the particles in. I will be using ``TS_43``. 
  
  - Click into the :bdg-secondary:`Pre-processing` block, go to **Tilt Series** tab and **Tomogram** sub tab. On this page, click the search icon, search for TS_43. Click the green button immediately above the tomogram display. This will download the tomogram in .rec format. 
  
  - Click into the :bdg-secondary:`Particle refinement` block, go to the **Metadata** tab. On this page, type ``TS_43`` into the search bar and click **Search**. Click the .star file to download particle alignments. 
  
  - Go to the **Reconstruction** tab and download the **Cropped Map**. 
    
.. nextpyp:: Step 2: Display in ChimeraX
  :collapsible: open

  - Open ChimeraX (again, we assume ArtiaX is installed)
  
  - Open the tomogram ``TS_43.rec``
  
  - Run the following commands in the ChimeraX shell:

  .. code-block:: bash

    volume permuteAxes #1 xzy
    volume flip #2 axis z
      
  - Go to the **ArtiaX** tab and click **Launch** to start the plugin. 
  
  - In the **Tomograms** section on the left, select model #3 (permuted z flip) from the **Add Model** dropdown menu and click **Add!**
  
  - Go to the ArtiaX options panel on the right, and set the **Pixel Size** for the **Current Tomogram** to 10.8 (The current binned pixel size) 
  
  - On the left panel, under the **Particles List** section, select **Open List ...** and open the .star file. 
  
  - Return to the panel on the right and select the **Select/Manipulate** tab. Set the **Origin** to 1.35 (the unbinned pixel size)
  
  - From the **Color Settings** section, select **Colormap** and then **rlnLogLikelihoodContribution** from the dropdown menu. 
  
  - Play with the **Marker Radius** and **Axes Size** sliders to visualize the particle locations, cross correlation scores, and orientations.


Session 2: On-the-fly pre-processing
====================================

Starting from **raw data** obtained at the microscope, we'll build an **automatic pipeline** that can perform all **pre-processing** tasks up to and including particle picking.

.. nextpyp:: Step 1: Creating/starting new sessions
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

.. nextpyp:: Step 2: Make changes and ``Restart`` ongoing sessions
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

.. nextpyp:: Step 3: Using the ``Clear`` option
  :collapsible: open

  * :bdg-primary:`Clear` will start pre-processing procedure from scratch

  * This is helpful if you want to start fresh making sure any previous pre-processing results are ignored.


.. nextpyp:: Step 4: Copying/deleting sessions
  :collapsible: open

  * Sessions can be **copied** or **deleted**.

  * Click the icon :fa:`location-arrow` to find the session's file storage location.

.. nextpyp:: Step 5: Importing/exporting sessions
  :collapsible: open

  Sessions can be exported in ``.star`` format for downstream processing and refinement in other software.

  * Navigate to the :bdg-Secondary:`Table` tab.

  * In the **Filters** box, type a name for your exported session.

  * Click :bdg-primary:`Export` to launch the export job. The job's log will indicate the location of the exported ``.star`` file.

.. nextpyp:: Step 6: Importing sessions into a projects
  :collapsible: open

  Since Sessions also perform pre-processing, we can import a finished Session into a project to kick-start the process of structure determination.

  * Click the :bdg-secondary:`Dashboard` link to go back to nextPYP's homepage.

  * Click the :bdg-primary:`Create New Project` button and give your project a name.

  * Click the :bdg-primary:`Import Data` button, and select the option :bdg-primary:`Tomography (from Session)`.

  * Search for the name of the session you wish to import.

  * Click the :bdg-primary:`Save` button, and then launch the job.

Day 4 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open
  
  * Continuously monitor raw data folder for incoming tilt-series

  * Raw data transfer (copy, link, move)

  * Pre-processing (frame alignment, tilt-series alignment, CTF estimation, and tomogram reconstruction)

  * Particle picking (geometry-based, size-based, etc.)

  * Restart, clear, copy or delete sessions

  * Import and export sessions

  Feel free to explore other options and functionality available in ``nextPYP`` as described in the :doc:`User Guide<../../guide/overview>`.