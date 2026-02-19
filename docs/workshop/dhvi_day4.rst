#####################
DHVI workshop (day 4)
#####################

Session 1: Map post-processing and visualization
================================================

Map post-processing
-------------------

.. nextpyp:: Step 1: Post-processing
  :collapsible: open
  
  * Click on ``Frames`` (output of the :bdg-secondary:`Movie refinement` block) and select :bdg-primary:`Post-processing`

  * Go to the **Post-processing** tab

    - Set the location of the ``First half map (*_half1.mrc)`` by selecting the file *"*_half1.mrc"*, then click :bdg-primary:`Choose File`

    - Set ``Masking method`` to "*from file*""

    - Set ``Mask file (*.mrc)`` to *"/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc"*, then click :bdg-primary:`Choose File`

    - Set ``B-factor method`` to *"adhoc"*

    - Set ``Adhoc value (A^2)`` to -25 

    - Set ``Min resolution (A)`` to 3

    - Set ``Max resolution (A)`` to 7

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step 2: Map and model assessment in ChimeraX
  :collapsible: open
  
  * We will be using a prealigned pdb file and files downloaded from nextPYP to demonstrate how one can visualize the final map aligned to a model in Chimera. 

  * Download files

    - In the :bdg-secondary:`Post-processing` block, go to the **Reconstruction** tab. Click on the drop down menu **Select an MRC file to download**. Select the **Full-Size Map** and the **Local Resolution Map**. You will download these files in MRC format. 

    - We are using a pre-aligned, pre-cropped pdb file (5L93) so do not need to download this. For your experiments, you would download whatever model required. 
  
    - Open both downloaded MRC files in Chimera. In the *Tools* menu, navigate to *Volume Data*, then *Surface Color*. In the *Surface Color* dialog, select to **color by** *volume data value*, and select the ``_resmap.mrc`` file in the **using map** field. Adjust the color values and press **Color**. Open an atomic model in Chimera. Under the **Map** tab, Click **Zone**. Note we are left with a slightly larger zone than we would like so we will copy the zone command from the output to the terminal line, and edit the range. This leaves us with: 

    .. code-block:: bash 

      volume zone #2 nearAtoms #1 range 2.4

    - Select the model, go to **Actions**, **Atoms/Bonds**, and **Show Sidechain/Base**
    
    - You can now view the model fit to your map interactively in ChimeraX


Visualization of results in ArtiaX/ChimeraX
-------------------------------------------

.. nextpyp:: Step 1: Download all the necessary files
  :collapsible: open

  - Select a tomogram you wish to visualize the particles in. I will be using ``TS_43``. 
  
  - Click into the :bdg-secondary:`Pre-processing` block, go to **Tilt Series** tab and **Tomogram** sub tab. On this page, click the search icon, search for ``TS_43``. Click the green button immediately above the tomogram display. This will download the tomogram in mrc format with the ``*.rec`` extension.
  
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

  .. figure:: ../images/guide_artiax_10164.webp
    :alt: ArtiaX visualization of HIV1-Gag

    Tomogram from immature HIV-1 virions from EMPIAR-10164 showing with a high-resolution model of Gag mapped back into the tomogram.


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

  * On the **Raw data** tab.

    * Set ``Path to raw data`` to *"/nfs/bartesaghilab/nextpyp/workshop/10164/TS_*.tif"*
  
  * On the **Microscope parameters** tab.

    - Set ``Pixel size (A)`` to 1.35

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 85.3
  
  * On the **Session settings** tab.

    - Set ``Number of tilts`` to 41

    - Set ``Raw data transfer`` to link (default)

      - ``Link``: Create a symlink between the data on the microscope and the Session folder. The data still *only* exists at the microscope.
      
      - ``Move``: Transfer the data from the microscope to the Session folder, removing the data at the microscope. The data will now *only* exist on the Sessions folder.
      
      - ``Copy``: Make a copy of the data in the microscope to your Session folder. The data will now exist at both the microscope *and* your Session folder.

  * On the **CTF Determination** tab.

    - Set ``Max resolution`` to 5
  
  * On the **Tomogram reconstruction** tab

    - Select the option ``Erase fiducials``

    - Select the option ``Generate half tomograms`` (you'll see why later...)

  * On the **Tomogram segmentation** tab

    - Select the option ``Pre-process tomograms``

      - Set ``Pixel size rescaling`` to 11

      - Enable ``Deconvolution filter``

    - Set ``Pre-trained model (*.ckpt)`` to *"/nfs/bartesaghilab/membrain-seg-models/MemBrain_seg_v10_alpha.ckpt"*

    - Set ``Filter connected components`` to *"by number"*

      - Set ``Largest components to ignore`` to 1

      - Set ``Components to keep`` to 16
    
    - Set ``Thickness of slab to keep (unbinned voxels)`` to 2048
    
    - Enable ``Test time augmentation``

    - Set ``Sliding window size`` to 96

  * On the **Virion detection** tab.

    - Set ``Virion radius`` to 500

    - Set ``Virion detection method`` to *"auto"*

    - Set ``Spike detection method`` to *"uniform"*

      - Set ``Size of equatorial band to restrict spike picking`` to 800
  
  * On the **Particle detection** tab.
  
    - Set ``Detection method`` to none

    - Set ``Detection radius`` to 50

  * On the **2D classification** tab.

    - Check the box for ``Show advanced options``

    - Check the box for ``Run 2D classification``

    - Set ``Incremental number of particles`` to 2000

    - Set ``Mask radius (A)`` to 300

    - Set ``Fraction of particles to use`` to 1
    
    - Set ``Starting high-resolution`` to 80

    - Set ``Max resolution (A)`` to 12 (Nyquist resolution plus some wiggle room)

  * On the **Resources** tab.
  
    - Set ``Split, Threads`` to 11

    - Set ``2D classification, Threads`` to 124 (we suggest using a number that matches the maximum allowed by your environment to see results more quickly)

    - Set ``Cluster Template`` to *NVIDIA A6000 Ada* (since membrain-seg needs GPUs to run)

  * Click :bdg-primary:`Save`, which will automatically take you to the :bdg-primary:`Operations` page.

  * Click :bdg-primary:`Start` to launch the session.

.. nextpyp:: Step 2: Make changes and ``Restart`` ongoing sessions
  :collapsible: open

  *  :bdg-primary:`Restart` is a "smart" method of re-running only what is necessary after changing pre-processing parameters.

  * Workflow: Change a parameter → :bdg-primary:`Save` settings changes → :bdg-primary:`Restart` pre-processing daemon.

  * Example: Adding on-the-fly tomogram denoising using IsoNet2

    * On the **Tomogram denoising** tab

      - Set ``Method`` to *isonet2*

      - Set the location of the ``Trained model`` to *"/nfs/bartesaghilab/nextpyp/workshop_dhvi/10164/isonet2-n2n_unet-medium_128_full_10A.pt"*

    * Click :bdg-primary:`Save`

    * Navigate to the :bdg-primary:`Operation` tab

    * Click :bdg-primary:`Restart` on the ``Data pre-processing`` daemon section

    * Open :bdg-primary:`Logs` to check that the restart flag has been detected and new pre-processing jobs will be launched in response to this change.

    * Check the ``Denoised`` tab, within the **Tilt Series** tab  to see the denoised tomograms.

.. nextpyp:: Step 3: Using the ``Clear`` option
  :collapsible: open

  * :bdg-primary:`Clear` will start pre-processing procedure from scratch

  * This is helpful if you want to start fresh making sure any previous pre-processing results are ignored.

  * Example: Changing the number of classes for 2D classification

    * In the **2D Classification** tab.

      - Set ``Number of classes`` to 10

    * Click :bdg-primary:`Save`
      
    * Navigate to the :bdg-primary:`Operation` tab

    * Click :bdg-primary:`Clear` on the ``2D classification`` daemon section

    * Open :bdg-primary:`Logs` to check that the clear flag has been detected and new 2D classification jobs will be launched in response to this change.

    * Check the **2D Classes** tab  to see your 10 new classes

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
  
  * How to continuously monitor raw data folder for incoming tilt-series

  * How to pre-process tilt-series on-the-fly (frame alignment, tilt-series alignment, CTF estimation, and tomogram reconstruction)

  * How to pick and segment virions, and pick particles during data collection

  * How to perform 2D projection classification to understand sample quality during data collection

  * How to Restart, Clear, Copy or Delete sessions

  * How to Export and Import sessions as projects

  This concludes the tutorial. The topics we covered this week represent a few key aspects of tomography data processing. Additional tools and functionality is described in the :doc:`User Guide<../guide/overview>`.