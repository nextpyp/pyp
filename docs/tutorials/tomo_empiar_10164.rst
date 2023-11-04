##################################
Tomography tutorial (EMPIAR-10164)
##################################

This tutorial shows how to convert raw tilt-series from `EMPIAR-10164 (HIV-1 Gag) <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_ into a ~3A resolution structure. 

Total running time required to complete this tutorial: ~20 hr.

We first use the command line to download and decompress a tbz file containing a subset of 5 tilt-series (down-sampled 2x compared to the original data), and an initial model:

.. code-block:: bash

  # cd to a location in the shared file system and run:

  wget https://nextpyp.app/files/data/nextpyp_tomo_tutorial.tbz
  tar xvfz nextpyp_tomo_tutorial.tbz

Open your browser and navigate to the url of your ``nextPYP`` instance (e.g., ``https://nextpyp.myorganization.org``).

Step 1: Create a new project
----------------------------

.. dropdown:: Data processing runs are organized into projects. We will create a new project for this tutorial
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * The first time you login into ``nextPYP``, you should see an empty **Dashboard**:

      .. figure:: ../images/dashboard_empty.webp
        :alt: Create new project

    * Click on :badge:`Create new project,badge-primary`, give the project a name, and select :badge:`Create,badge-primary`

      .. figure:: ../images/tutorial_tomo_new.webp
        :alt: Create new project

    * Select the new project from the **Dashboard** and click :badge:`Open,badge-primary`

      .. figure:: ../images/tutorial_tomo_open.webp
        :alt: Select new project

    * The newly created project will be empty and a **Jobs** panel will appear on the right

      .. figure:: ../images/tutorial_tomo_empty.webp
        :alt: Empty project

Step 2: Import raw tilt-series
------------------------------

.. dropdown:: Import the raw tilt-series downloaded above (:fa:`stopwatch` <1 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Go to :badge:`Import Data,badge-primary` and select :badge:`Tomography (from Raw Data),badge-primary`

      .. figure:: ../images/tutorial_tomo_import_dialog.webp
        :alt: Import dialog

    * A form to enter parameters will appear:

      .. figure:: ../images/tutorial_tomo_import_data.webp
        :alt: File browser

    * Go to the **Raw data** tab:

      .. tabbed:: Raw data

        - Set the ``Location`` of the raw data by clicking on the icon :fa:`search,text-primary` and browsing to the directory where the you downloaded the raw movie frames

        - Type ``TS_*.tif`` in the filter box (lower right) and click on the icon :fa:`filter,text-primary` to verify your selection. 205 matches should be displayed

        - Click :badge:`Choose File Pattern,badge-primary` to save your selection

        - Click on the **Microscope parameters** tab

        .. figure:: ../images/tutorial_tomo_import_browser.webp
          :alt: File browser

      .. tabbed:: Microscope parameters

        - Set ``Pixel size (A)`` to 1.35

        - Set ``Acceleration voltage (kV)`` to 300

        - Set ``Tilt-axis angle (degrees)`` to 85.3

        .. figure:: ../images/tutorial_tomo_microscope_params.webp
          :alt: Project dashboard

    * Click :badge:`Save,badge-primary` and the new block will appear on the project page

      .. figure:: ../images/tutorial_tomo_import_modified.webp
        :alt: Project dashboard

    * The block is in the modified state (indicated by the :fa:`asterisk` sign) and is ready to be executed

    * Clicking the button :badge:`Run,badge-primary` will show another dialog where you can select which blocks to run:

      .. figure:: ../images/tutorial_tomo_import_run_dialog.webp
        :alt: Gain thumbnail

    * Since there is only one block available, simply click on :badge:`Start Run for 1 block,badge-primary`. This will launch a process that reads one tilt image, applies the gain reference (if applicable) and displays the resulting image inside the block

      .. figure:: ../images/tutorial_tomo_import_done.webp
        :alt: Gain thumbnail

    * Click inside the block to see a larger version of the image


Step 3: Pre-processing
----------------------

.. dropdown:: Movie frame alignment, CTF estimation and particle picking (:fa:`stopwatch` 6 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Tilt-series` (output of the :badge:`Tomography (from Raw Data),badge-secondary` block) and select :badge:`Pre-processing,badge-primary`

      .. figure:: ../images/tutorial_tomo_pre_process_dialog.webp
        :alt: File browser

    * Go to the **Frame alignment** tab:

      .. tabbed:: Frame alignment

        - Set ``Frame pattern`` to TILTSERIES_SCANORD_ANGLE.tif

        - Click on the **CTF determination** tab

      .. tabbed:: CTF determination

        - Set ``Max resolution`` to 5.0

        - Click on the **Virion/spike detection** tab

      .. tabbed:: Virion/spike detection

        - Set ``Virion detection method`` to auto

        - Set ``Virion radius (A)`` to 500.0

        - Click on the **Tomogram reconstruction** tab

      .. tabbed:: Tomogram reconstruction
        
        - Click ``Show advanced options``

        - Set ``Binning factor for reconstruction`` to 8

        - Set ``Thickness of reconstruction (unbinned voxels)`` to 2048

        - Click on the **Resources** tab
      
      .. tabbed:: Resources

        - Set ``Threads per task`` to 7

        - Set ``Memory per task`` to 14

        - Set other runtime parameters as needed (see :doc:`Computing resources<../reference/computing>`)

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. Follow the status of the run in the **Jobs** panel

      .. figure:: ../images/tutorial_tomo_pre_process_modified.webp
        :alt: File browser

    * Click inside the :badge:`Pre-processing,badge-secondary` block to inspect the results (you don't need to wait until processing is done to do this). Results will be grouped into tabs:

      .. tabbed:: Plots

        .. figure:: ../images/tutorial_tomo_pre_process_page.webp
          :alt: Dataset statistics

      .. tabbed:: Table

        .. figure:: ../images/tutorial_tomo_pre_process_table.webp
          :alt: Table view

      .. tabbed:: Gallery

        .. figure:: ../images/tutorial_tomo_pre_process_gallery.webp
          :alt: Gallery view

      .. tabbed:: Tilt-series

        .. tabbed:: Tilts

          .. figure:: ../images/tutorial_tomo_pre_process_tilts.webp
            :alt: Tilt-series (Tilts)

        .. tabbed:: Alignment

          .. figure:: ../images/tutorial_tomo_pre_process_alignments.webp
            :alt: Tilt-series (Alignment)

        .. tabbed:: CTF

          .. figure:: ../images/tutorial_tomo_pre_process_ctf.webp
            :alt: Tilt-series (CTF)

        .. tabbed:: Reconstruction

          .. figure:: ../images/tutorial_tomo_pre_process_reconstruction.webp
            :alt: Tilt-series (Reconstruction)

        .. tabbed:: Segmentation

          .. figure:: ../images/tutorial_tomo_pre_process_segmentation.webp
            :alt: Tilt-series (Segmentation)

.. tip::

  While on the **Tilt Series** tab, use the navigation bar at the top of the page to look at the results for other tilt-series

Step 4 (optional): Virion segmentation
--------------------------------------

.. dropdown:: Detection and segmentation of virions (interactive step)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    This step is optional, but it showcases tools available in ``nextPYP`` to work with virions:

    * Go inside the :badge:`Pre-processing,badge-secondary` block, click on the **Tilt Series** tab, and select :badge:`Segmentation,badge-primary`

      .. figure:: ../images/tutorial_tomo_pre_process_virions.webp
        :alt: Virion segmentation

    * Select a virion from the table to show its 3D segmentation (8 different thresholds are shown as yellow contours in columns 1-8). The column number highlighted in blue represents the selected threshold value (default is 1, click on a different column to select a better threshold). If none of the columns look reasonable (or if you want to ignore the current virion virion), select the last column ("-")

    * Repeat this process for all virions in the table and all tilt-series in the dataset

.. tip::

  Click on `> Keyboard shortcuts` (under the virion image) to reveal instructions on how to speed up the threshold selection process

Step 5: Particle detection
--------------------------

.. dropdown:: Particle detection from virion surfaces (:fa:`stopwatch` 3 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * In the :badge:`Pre-processing,badge-primary` block, click on the menu icon :fa:`bars, text-primary` and select the :fa:`edit, text-primary` Edit option.

    * Go to the **Virion/spike detection** tab:

      .. tabbed:: Virion/spike detection

        - Set ``Spike detection method`` to uniform

        - Set ``Minimum distance between spikes (voxels)`` to 8

        - Set ``Size of equatorial band to restrict spike picking (A)`` to 800

        - Click on the **Particle detection** tab

      .. tabbed:: Particle detection

        - Set ``Detection radius (A)`` to 50

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Navigate to the :badge:`Reconstruction,badge-primary` group inside the **Tilt-series** tab to inspect the virion and spike coordinates:

      .. figure:: ../images/tutorial_tomo_pre_process_spikes.webp
        :alt: Spike coordinates

Step 6: Reference-based refinement
----------------------------------

.. dropdown:: Reference-based particle alignment (:fa:`stopwatch` 8 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle refinement,badge-primary`

    * Go to the **Sample** tab:

      .. tabbed:: Sample

        - Set ``Molecular weight (kDa)`` to 300

        - Set ``Particle radius (A)`` to 150

        - Set ``Symmetry`` to C6

        - Click on the **Extraction** tab

      .. tabbed:: Extraction

        - Set ``Box size (pixels)`` to 192

        - Set ``Image binning`` to 2

        - Click on the **Refinement** tab

      .. tabbed:: Refinement

        - Specify the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary`, navigating to the folder where you downloaded the data for the tutorial, and selecting the file `EMPIAR-10164_init_ref.mrc`

        - Click :fa:`search,text-primary` in ``Input parameter file`` to select the initial parameters text file ``tomo-preprocessing-*_original_volumes.txt`` from :badge:`Pre-processing,badge-secondary`

        - Check ``Skip refinement``

        - Set ``Max resolution (A)`` to 8.0

        - Check ``Use signed correlation``

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``Last exposure for refinement`` to 10

        - Check ``Show advanced options``

        - Set ``Optimizer - Exhaustive search points`` to 50000

        - Check ``Refine particle alignments``

        - Set ``Particle Phi range (degrees)`` and ``Particle Theta range (degrees)`` to 10

        - Set ``Particle translation range (voxels)`` to 50

        - Click on the **Resources** tab
  
      .. tabbed:: Reconstruction
        
        - Set ``Max tilt-angle`` to 50

        - Set ``Min tilt-angle`` to -50
      
      .. tabbed:: Resources

        - Set ``Walltime per task`` to 9:00:00

        - Set ``Threads (merge task)`` to 6

        - Set ``Memory (merge task)`` to 20

    * :badge:`Save,badge-primary` your changes, click :badge:`Run,badge-primary` and :badge:`Start Run for 1 block,badge-primary`

    * One round of refinement and reconstruction will be executed. Click inside the block to see the results

      .. figure:: ../images/tutorial_tomo_coarse_iter2.webp
        :alt: Iter 2


Step 7. Fully constrained refinement
------------------------------------

.. dropdown:: Tilt-geometry parameters and particle poses are refined in this step (:fa:`stopwatch` 1.5 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Edit the settings of the existing :badge:`Particle refinement,badge-secondary` block and go the **Refinement** tab:

      .. tabbed:: Refinement

        - Set ``Last iteration`` to 5

        - Set ``Max resolution (A)`` to 8:10:8:6

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Check ``Refine tilt-geometry``

        - Check ``Show advanced options``

        - Set ``Optimizer - Max step length`` to 100

        - Set ``Optimizer - Exhaustive search points`` to 0

        - Set ``Particle Phi range``, ``Particle Psi range`` and ``Particle Theta range`` to 20.0

        - Click on the **Exposure weighting** tab

      .. tabbed:: Exposure weighting

        - Check ``Dose weighting``
        
        - Set ``Frame weight fraction`` to 4

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to execute three rounds of refinement and reconstruction

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results:

      .. figure:: ../images/tutorial_tomo_coarse_iter5.webp
        :alt: Iter 5

.. tip::

  Use the navigation bar at the top left of the page to look at the results for different iterations

Step 8. Filter particles
------------------------

.. dropdown:: Identify duplicates and particles with low alignment scores (:fa:`stopwatch` 4 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of the :badge:`Particle refinement,badge-secondary` block) and select :badge:`Filter particles,badge-primary`

    * Go to the **Particle filtering** tab:

      .. tabbed:: Particle filtering

        - Set ``Score threshold`` to 2.5

        - Set ``Min distance between particles (A)`` to 10

        - Specify the location of ``Input parameter file`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_05.par.bz2`

        - Set ``Lowest tilt-angle`` to -15.0

        - Set ``Highest tilt-angle`` to 15.0

        - Check ``Generate reconstruction after filtering``

        - Click on the **Refinement** tab

      .. tabbed:: Refinement

        - Specify the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_05.mrc`

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. You can see how many particles were left after filtering by looking at the job logs.

Step 9 (optional): Permanently remove bad particles
---------------------------------------------------

.. dropdown:: Permanently remove bad particles to improve processing efficiency downstream (:fa:`stopwatch` 1 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Edit the settings of the existing :badge:`Filter particles,badge-secondary` block

    * Go to the **Particle refinement** tab:

      .. tabbed:: Particle filtering

        - Check ``Permanently remove particles``

        - Uncheck ``Generate reconstruction after filtering``

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to launch the job


Step 10. Region-based local refinement (before masking)
-------------------------------------------------------

.. dropdown:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 1 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of :badge:`Filter particles,badge-secondary` block) and select :badge:`Particle refinement,badge-primary`

    * Go to the **Sample** tab:

      .. tabbed:: Sample

        - Set ``Particle radius`` to 100

        - Click on the **Extraction** tab

      .. tabbed:: Extraction

        - Set ``Box size (pixels)`` to 384

        - Set ``Image binning`` to 1

        - Click on the **Refinement** tab

      .. tabbed:: Refinement

        - Specify the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-fine-refinement-*_r01_02.mrc`

        - Select the location of the ``Initial parameter file`` by clicking on the icon :fa:`search,text-primary` and selecting the file `tomo-fine-refinement-*_r01_02.par.bz2` (select the file ``tomo-fine-refinement-*_r01_02_clean.par.bz2`` if bad particles were permanently removed in the previous step)

        - Set ``Last iteration`` to 3

        - Check ``Skip refinement``

        - Set ``Max resolution (A)`` to 6:5

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``Last exposure for refinement`` to 4

        - Set ``Number of regions`` to 8,8,2

        - Check ``Show advanced options``

        - Set ``Particle translation range (voxels)`` to 20.0

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to run the job

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results:

      .. figure:: ../images/tutorial_tomo_region_before_masking_iter3.webp
        :alt: Iter 3


Step 11: Create shape mask
--------------------------

.. dropdown:: Use most recent reconstruction to create a shape mask (:fa:`stopwatch` <1 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of :badge:`Particle refinement,badge-secondary` block) and select :badge:`Masking,badge-primary`

    * Go to the **Masking** tab:

      .. tabbed:: Masking

        - Select the ``Input map`` by click on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_03.mrc`

        - Set ``Threshold for binarization`` to 0.45

        - Check ``Use normalized threshold``

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to run the job

    * Click on the menu icon :fa:`bars, text-primary` of the :badge:`Masking,badge-secondary` block, select the :badge:`Show Filesystem Location` option, and :badge:`Copy,badge-primary` the location of the block in the filesystem (we will use this in the next step))

    * Click inside the :badge:`Masking,badge-secondary` block to inspect the results of masking

Step 12: Region-based constrained refinement
--------------------------------------------

.. dropdown:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 2 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Edit the settings of the existing :badge:`Particle refinement,badge-secondary` block and go to the **Refinement** tab:

      .. tabbed:: Refinement

        - Set ``First iteration`` to 4

        - Set ``Last iteration`` to 6

        - Set ``Max resolution (A)`` to 6:5:5:4:3.5

        - Specify the location of the ``Shape mask`` produced in Step 11 by clicking on the icon :fa:`search, text-primary`, navigating to the location of the :badge:`Masking,badge-secondary` block by copying the path we saved above, and selecting the file `frealign/maps/mask.mrc`

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to run the job

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results:

      .. figure:: ../images/tutorial_tomo_region_iter6.webp
        :alt: Iter 6

Step 13: Particle-based CTF refinement
--------------------------------------

.. dropdown:: Per-particle CTF refinement using most recent reconstruction (:fa:`stopwatch` 3 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on the menu icon :fa:`bars, text-primary` from the :badge:`Particle refinement,badge-secondary` block and choose the :fa:`edit, text-primary` Edit option

    * Go to the **Refinement** tab:

      .. tabbed:: Refinement

        - Set ``Last iteration`` to 7

        - Set ``Max resolution (A)`` to 3.1

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Uncheck ``Refine tilt-geometry``

        - Uncheck ``Refine particle alignments``

        - Check ``Refine CTF per-particle``

        - Check ``Show advanced options``

        - Set ``Last exposure for refinement`` to 10

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results

      .. figure:: ../images/tutorial_tomo_ctf_iter7.webp
        :alt: Iter 7

Step 14: Movie frame refinement
-------------------------------

.. dropdown:: Particle-based movie-frame alignment and data-driven exposure weighting (:fa:`stopwatch` 3 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click :guilabel:`Particles` (output of :badge:`Particle refinement,badge-secondary` block) and select :badge:`Movie refinement,badge-primary`

    * Go to the **Sample** tab:

      .. tabbed:: Sample

        - Set ``Particle radius`` to 80

        - Click on the **Refinement** tab

      .. tabbed:: Refinement

        - Specify the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_07.mrc`

        - Specify the ``Input parameter file`` by clicking on the icon :fa:`search,text-primary` and selecting the file `tomo-coarse-refinement-*_r01_07.par.bz2`

        - Set ``Max resolution (A)`` to 3.2

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Uncheck ``Refine CTF per-particle``

        - Check ``Movie frame refinement``

        - Check ``Show advanced options``

        - Set ``Last exposure for refinement`` to 4

        - Check ``Regularize translations``

        - Set ``Spatial sigma`` to 200.0

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results:

      .. figure:: ../images/tutorial_tomo_movie_iter2.webp
        :alt: Iter 2

Step 15: Refinement after movie frame refinement
------------------------------------------------

.. dropdown:: Additional refinement using new frame alignment parameters (:fa:`stopwatch` 1 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on the menu icon :fa:`bars, text-primary` from the :badge:`Movie refinement,badge-secondary` block and choose the :fa:`edit, text-primary` Edit option.

    * Go to the **Refinement** tab:
        
      .. tabbed:: Refinement

        - Set ``Max resolution (A)`` to 3.3

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Check ``Refine tilt-geometry``

        - Check ``Refine particle alignments``

        - Uncheck ``Movie frame refinement``
        
        - Set ``Micrograph translation range (pixels)`` to 10.0

        - Set ``Micrograph tilt-angle range (degrees)`` and ``Micrograph tilt-axis range (degrees)`` to 1.0

        - Set ``Particle Phi range (degrees)`` to 1.0

        - Set ``Particle Psi range (degrees)`` to 1.0

        - Set ``Particle Theta range (degrees)`` to 1.0

        - Set ``Particle translation range (voxels)`` to 10.0
        
        - Set ``Min number of projections for refinement`` to 2

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results:

      .. figure:: ../images/tutorial_tomo_after_movie_iter3.webp
        :alt: Iter 3

Step 16: Map sharpening
-----------------------

.. dropdown:: Apply B-factor weighting in frequency space (:fa:`stopwatch` <1 min)
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    * Click :guilabel:`Movies` (output of :badge:`Movie refinement,badge-secondary` block) and select :badge:`Post-processing,badge-primary`

    * Go to the **Post-processing** tab:

      .. tabbed:: Post-processing

       - Specify the ``First half map`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-flexible-refinement-*_r01_half1.mrc` (output of the :badge:`Movie refinement,badge-secondary` block)

       - Set ``Automask threshold`` to 0.4

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * You can inspect the result by clicking inside the :badge:`Map sharpening,badge-secondary` block:

      .. figure:: ../images/tutorial_tomo_final_map.webp
        :alt: Final map

.. note::

  Running times were measured running all tilt-series in parallel on nodes with 124 vCPUs, 720GB RAM, and 3TB of local SSDs

.. seealso::

    * :doc:`Classification tutorial<tomo_empiar_10304>`
    * :doc:`Single-particle tutorial<spa_empiar_10025>`
    * :doc:`Single-particle session<stream_spr>`
    * :doc:`Tomography session<stream_tomo>`
