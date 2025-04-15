##################################
Tomography tutorial (EMPIAR-10164)
##################################

This tutorial shows how to convert raw tilt-series from `EMPIAR-10164 <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_ into a ~3A resolution structure of immature HIV-1 Gag protein.

.. admonition::

  * Total running time required to complete this tutorial: **~20 hrs**.

  * Pre-calculated results are available in `the demo instance of nextPYP <https://demo.nextpyp.app/#/project/ab690@duke.edu/EMPIAR-10164-RtQMJrzN90C81gbU>`_.

We first use the command line to download and decompress a `.tbz file <https://nextpyp.app/files/data/nextpyp_tomo_tutorial.tbz>`_ containing a subset of 5 tilt-series (down-sampled 2x compared to the original super-resolution data):

.. code-block:: bash

  # cd to a location in the shared file system and run:

  wget https://nextpyp.app/files/data/nextpyp_tomo_tutorial.tbz
  tar xvfz nextpyp_tomo_tutorial.tbz

After this, you should have 41 tilt movies in .tif format for each of the tilt-series (TS_01, TS_03, TS_43, TS_45, and TS_54), an initial reference, and a shape mask

Open your browser and navigate to the url of your ``nextPYP`` instance (e.g., ``https://nextpyp.myorganization.org``).

Step 1: Create a new project
----------------------------

.. nextpyp:: Data processing runs are organized into projects. We will create a new project for this tutorial
  :collapsible: open

  * The first time you login into ``nextPYP``, you should see an empty **Dashboard**:

    .. figure:: ../images/dashboard_empty.webp
      :alt: Create new project

  * Click on :bdg-primary:`Create new project`, give the project a name, and select :bdg-primary:`Create`

    .. figure:: ../images/tutorial_tomo_new.webp
      :alt: Create new project

  * Select the new project from the **Dashboard** and click :bdg-primary:`Open`

    .. figure:: ../images/tutorial_tomo_open.webp
      :alt: Select new project

  * The newly created project will be empty and a **Jobs** panel will appear on the right

    .. figure:: ../images/tutorial_tomo_empty.webp
      :alt: Empty project

Step 2: Import raw tilt-series
------------------------------

.. nextpyp:: Import the raw tilt-series downloaded above (:fa:`stopwatch` <1 min)
  :collapsible: open

  * Go to :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`

    .. figure:: ../images/tutorial_tomo_import_dialog.webp
      :alt: Import dialog

  * A form to enter parameters will appear:

    .. figure:: ../images/tutorial_tomo_import_data.webp
      :alt: File browser

  * Go to the **Raw data** tab:

    .. md-tab-set::

      .. md-tab-item:: Raw data

        - Set the ``Location`` of the raw data by clicking on the icon :fa:`search` and browsing to the directory where the you downloaded the raw movie frames

        - Type ``TS_*.tif`` in the filter box (lower right) and click on the icon :fa:`filter` to verify your selection. 205 matches should be displayed

        - Click :bdg-primary:`Choose File Pattern` to save your selection

        - Click on the **Microscope parameters** tab

        .. figure:: ../images/tutorial_tomo_import_browser.webp
          :alt: File browser

      .. md-tab-item:: Microscope parameters

        - Set ``Pixel size (A)`` to 1.35

        - Set ``Acceleration voltage (kV)`` to 300

        - Set ``Tilt-axis angle (degrees)`` to 85.3

        .. figure:: ../images/tutorial_tomo_microscope_params.webp
          :alt: Project dashboard

  * Click :bdg-primary:`Save` and the new block will appear on the project page

    .. figure:: ../images/tutorial_tomo_import_modified.webp
      :alt: Project dashboard

  * The block is in the modified state (indicated by the :fa:`asterisk` sign) and is ready to be executed

  * Clicking the button :bdg-primary:`Run` will show another dialog where you can select which blocks to run:

    .. figure:: ../images/tutorial_tomo_import_run_dialog.webp
      :alt: Gain thumbnail

  * Since there is only one block available, simply click on :bdg-primary:`Start Run for 1 block`. This will launch a process that reads one tilt at random and displays the resulting image inside the block

    .. figure:: ../images/tutorial_tomo_import_done.webp
      :alt: Gain thumbnail

  * Click on the thumbnail inside the block to see a larger version of the projection image


Step 3: Pre-processing
----------------------

.. nextpyp:: Movie frame alignment, and CTF estimation (:fa:`stopwatch` 5 min)
  :collapsible: open

  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

    .. figure:: ../images/tutorial_tomo_pre_process_dialog.webp
      :alt: File browser

  * Go to the **Frame alignment** tab:

    .. md-tab-set::

      .. md-tab-item:: Frame alignment

        - Leave ``Frame pattern`` as the default value *TILTSERIES_SCANORD_ANGLE.tif*. ``nextPYP`` uses this to extract the metadata from the file names, for example, ``TS_54_037_57.0.tif`` would indicate that the tilt-series name is ``TS_54``, the exposure acquistion order is ``37``, and the corresponding tilt-angle is ``57.0`` degrees

        - Click on the **CTF determination** tab

      .. md-tab-item:: CTF determination

        - Set ``Max resolution`` to 5.0

        - Click on the **Resources** tab

      .. md-tab-item:: Resources

        - Set ``Threads per task`` to 11

        - Set other runtime parameters as needed (see :doc:`Computing resources<../reference/computing>`)

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

    .. figure:: ../images/tutorial_tomo_pre_process_modified.webp
      :alt: File browser

  * Click inside the :bdg-secondary:`Pre-processing` block to inspect the results (you don't need to wait until processing is done to do this). Results will be grouped into tabs:

    .. md-tab-set::

      .. md-tab-item:: Plots

        .. figure:: ../images/tutorial_tomo_pre_process_page.webp
          :alt: Dataset statistics

      .. md-tab-item:: Table

        .. figure:: ../images/tutorial_tomo_pre_process_table.webp
          :alt: Table view

      .. md-tab-item:: Gallery

        .. figure:: ../images/tutorial_tomo_pre_process_gallery.webp
          :alt: Gallery view

      .. md-tab-item:: Tilt-series

        .. md-tab-set::
          
          .. md-tab-item:: Tilts

            .. figure:: ../images/tutorial_tomo_pre_process_tilts.webp
              :alt: Tilt-series (Tilts)

          .. md-tab-item:: Alignment

            .. figure:: ../images/tutorial_tomo_pre_process_alignments.webp
              :alt: Tilt-series (Alignment)

          .. md-tab-item:: CTF

            .. figure:: ../images/tutorial_tomo_pre_process_ctf.webp
              :alt: Tilt-series (CTF)

          .. md-tab-item:: Reconstruction

            .. figure:: ../images/tutorial_tomo_pre_process_reconstruction.webp
              :alt: Tilt-series (Reconstruction)

    .. tip::

      While on the **Tilt Series** tab, use the navigation bar at the top of the page to look at the results for other tilt-series

Step 4: Virion selection
------------------------

.. nextpyp:: Selection of virion centers (:fa:`stopwatch` 1 min)
  :collapsible: open

  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle picking`

  * Go to the **Particle detection** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle detection

        - Set ``Detection method`` to virions

        - Set ``Virion radius (A)`` to 500

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

Step 5: Virion segmentation
---------------------------

.. nextpyp:: Segment individual virions in 3D (:fa:`stopwatch` 1 min)
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

  This next step is optional, but it showcases tools available in ``nextPYP`` to work with virions:

  * Go inside the :bdg-secondary:`Segmentation (closed surfaces)` block and click on the **Segmentation** tab

    .. figure:: ../images/tutorial_tomo_pre_process_virions.webp
      :alt: Virion segmentation

  * Select a virion from the table to show its 3D segmentation (8 different thresholds are shown as yellow contours in columns 1-8). The column number highlighted in blue represents the selected threshold value (default is 1, click on a different column to select a better threshold). The best threshold is the one that more closely follows the outermost membrane layer. If none of the columns look reasonable (or if you want to ignore the current virion), select the last column ("-")

  * Repeat this process for all virions in the table and all tilt-series in the dataset

    .. tip::

      Click on `> Keyboard shortcuts` (under the virion image) to reveal instructions on how to speed up the threshold selection process

Step 6: Particle picking
------------------------

.. nextpyp:: Select particles from the surface of virions (:fa:`stopwatch` 3 min)
  :collapsible: open

  * Click on ``Segmentation (closed)`` (output of the :bdg-secondary:`Segmentation (closed surfaces)` block) and select :bdg-primary:`Particle picking (closed surfaces)`

  * Go to the **Particle detection** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle detection

        - Set ``Particle radius (A)`` to 50

        - Set ``Detection method`` to uniform

        - Set ``Minimum distance between particles (voxels)`` to 8

        - Set ``Size of equatorial band to restrict search (A)`` to 800

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

  * Navigate to the :bdg-primary:`Reconstruction` tab to inspect the particle coordinates:

    .. figure:: ../images/tutorial_tomo_pre_process_spikes.webp
      :alt: Spike coordinates

Step 7: Reference-based refinement
----------------------------------

.. nextpyp:: Constrained reference-based particle alignment (:fa:`stopwatch` 8 hr)
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle picking (closed surfaces)` block) and select :bdg-primary:`Reference-based refinement`

  * Go to the **Sample** tab:

    .. md-tab-set::

      .. md-tab-item:: Sample

        - Set ``Molecular weight (kDa)`` to 300

        - Set ``Particle radius (A)`` to 150

        - Set ``Symmetry`` to C6

        - Click on the **Particle extraction** tab

      .. md-tab-item:: Particle extraction

        - Set ``Box size (pixels)`` to 192

        - Set ``Image binning`` to 2

        - Click on the **Particle scoring function** tab

      .. md-tab-item:: Particle scoring function

        - Set ``Last tilt for refinement`` to 10

        - Set ``Max resolution (A)`` to 8.0

        - Click on the **Reference-based refinement** tab

      .. md-tab-item:: Reference-based refinement

        - Specify the location of the ``Initial model (*.mrc)`` by clicking on the icon :fa:`search`, navigating to the folder where you downloaded the data for the tutorial, and selecting the file `EMPIAR-10164_init_ref.mrc`

        - Set ``Particle rotation Psi range (degrees)`` and ``Particle rotation Theta range (degrees)`` to 10

        - Set ``Particle rotation step (degrees)`` to 2

        - Set ``Particle translation range (A)`` to 50

        - Set ``Particle translation step (A)`` to 6

        - Click on the **Reconstruction** tab

      .. md-tab-item:: Reconstruction

        - Check ``Show advanced options``

        - Set ``Max tilt-angle`` to 50

        - Set ``Min tilt-angle`` to -50

        - Click on the **Resources** tab

      .. md-tab-item:: Resources

        - Set ``Threads per task`` to the maximum allowable by your system

        - Set ``Threads (merge task)`` to 6

  * :bdg-primary:`Save` your changes, click :bdg-primary:`Run` and :bdg-primary:`Start Run for 1 block`

  * One round of refinement and reconstruction will be executed. Click inside the block to see the results

    .. figure:: ../images/tutorial_tomo_coarse_iter2.webp
      :alt: Iter 2


Step 8. 3D refinement
---------------------

.. nextpyp:: Tilt-geometry parameters and particle poses are refined in this step (:fa:`stopwatch` 1.5 hr)
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Reference-based refinement` block) and select :bdg-primary:`3D refinement`

  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

        - Set ``Max resolution (A)`` to 10:8:6 (this will use an 10A limit for the first iteration, 8A for the second, etc.)

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Specify the location of ``Input parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file `tomo-reference-refinement-*_r01_02.bz2`

        - Set ``Last iteration`` to 4

        - Check ``Refine tilt-geometry``

        - Check ``Refine particle alignments``

        - Set ``Particle rotation Phi range (degrees)``, ``Particle rotation Psi range (degrees)`` and ``Particle rotation Theta range (degrees)`` to 20.0

        - Set ``Particle translation range (A)`` to 100
 
        - Check ``Show advanced options``

        - Set ``Optimizer - Max step length`` to 100

        - Click on the **Reconstruction** tab

      .. md-tab-item:: Reconstruction

        - Check ``Apply dose weighting``
        
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to execute three rounds of refinement and reconstruction

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results:

    .. figure:: ../images/tutorial_tomo_coarse_iter5.webp
      :alt: Iter 5

    .. tip::

      Use the navigation bar at the top left of the page to look at the results for different iterations

Step 9. Filter particles
------------------------

.. nextpyp:: Identify and remove duplicates and particles with low alignment scores (:fa:`stopwatch` 4 min)
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`3D refinement` block) and select :bdg-primary:`Particle filtering`

  * Go to the **Particle filtering** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle filtering

        - Specify the location of ``Input parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file `tomo-new-coarse-refinement-*_r01_04.bz2`

        - Set ``Score threshold`` to 2.5

        - Set ``Min distance between particles (unbinned pixels/voxels)`` to 10

        - Set ``Lowest tilt-angle (degrees)`` to -15

        - Set ``Highest tilt-angle (degrees)`` to 15

        - Check ``Generate reconstruction after filtering``

        - Check ``Permanently remove particles``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. You can see how many particles were left after filtering by looking at the job logs.

Step 10. Region-based local refinement (before masking)
-------------------------------------------------------

.. nextpyp:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 1 hr)
  :collapsible: open

  * Click on ``Particles`` (output of :bdg-secondary:`Particle filtering` block) and select :bdg-primary:`3D refinement`

  * Go to the **Sample** tab:

    .. md-tab-set::

      .. md-tab-item:: Sample

        - Set ``Particle radius`` to 100

        - Click on the **Particle extraction** tab

      .. md-tab-item:: Particle extraction

        - Set ``Box size (pixels)`` to 384

        - Set ``Image binning`` to 1

        - Click on the **Particle scoring function** tab

      .. md-tab-item:: Particle scoring function

        - Set ``Last tilt for refinement`` to 4

        - Set ``Max resolution (A)`` to 6:5

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Select the location of the ``Initial parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file ``tomo-fine-refinement-*_r02_clean.bz2``

        - Set ``Last iteration`` to 3

        - Set ``Number of regions`` to 8,8,2

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to run the job

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results:

    .. figure:: ../images/tutorial_tomo_region_before_masking_iter3.webp
      :alt: Iter 3


Step 11: Create shape mask
--------------------------

.. nextpyp:: Use most recent reconstruction to create a shape mask (:fa:`stopwatch` <1 min)
  :collapsible: open

  * Click on ``Particles`` (output of the last :bdg-secondary:`3D refinement` block) and select :bdg-primary:`Masking`

  * Go to the **Masking** tab:

    .. md-tab-set::

      .. md-tab-item:: Masking

        - Select the ``Input map (*.mrc)`` by click on the icon :fa:`search` and selecting the file `tomo-new-coarse-refinement-*_r01_03.mrc`

        - Set ``Threshold for binarization`` to 0.45

        - Set ``Width of cosine edge (pixels)`` to 8

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to run the job

  * Click on the menu icon :fa:`bars` of the :bdg-secondary:`Masking` block, select the :bdg-secondary:`Show Filesystem Location` option, and :bdg-primary:`Copy` the location of the block in the filesystem (we will use this in the next step))

  * Click inside the :bdg-secondary:`Masking` block to inspect the results of masking.

  .. note::

    You may need to adjust the binarization threshold to obtain a mask that includes the protein density and excludes the background (a pre-calculated mask is provided with the raw data if you rather use that).

Step 12: Region-based constrained refinement
--------------------------------------------

.. nextpyp:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 2 hr)
  :collapsible: open

  * Click on ``Particles`` (output of the last :bdg-secondary:`3D refinement` block) and select :bdg-primary:`3D refinement`
  
  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

        - Set ``Max resolution (A)`` to 5:4:3.5

        - Set ``Masking strategy`` to from file
        
        - Specify the location of the ``Shape mask`` produced in Step 11 by clicking on the icon :fa:`search`, navigating to the location of the :bdg-secondary:`Masking` block by copying the path we saved above, and selecting the file `frealign/maps/mask.mrc`

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Select the ``Input parameter file (*.bz2)`` by click on the icon :fa:`search` and selecting the file `tomo-new-coarse-refinement-*_r01_03.bz2`

        - Set ``Last iteration`` to 4

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to run the job

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results:

    .. figure:: ../images/tutorial_tomo_region_iter6.webp
      :alt: Iter 6

Step 13: Particle-based CTF refinement
--------------------------------------

.. nextpyp:: Per-particle CTF refinement using most recent reconstruction (:fa:`stopwatch` 3 hr)
  :collapsible: open

  * Click on ``Particles`` (output of the last :bdg-secondary:`3D refinement` block) and select :bdg-primary:`3D refinement`
  
  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

        - Set ``Last tilt for refinement`` to 10

        - Set ``Max resolution (A)`` to 3.1

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Select the ``Input parameter file (*.bz2)`` by click on the icon :fa:`search` and selecting the file `tomo-new-coarse-refinement-*_r01_02.bz2`

        - Uncheck ``Refine tilt-geometry``

        - Uncheck ``Refine particle alignments``

        - Check ``Refine CTF per-particle``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results

    .. figure:: ../images/tutorial_tomo_ctf_iter7.webp
      :alt: Iter 7

Step 14: Movie frame refinement
-------------------------------

.. nextpyp:: Particle-based movie-frame alignment and data-driven exposure weighting (:fa:`stopwatch` 3 hr)
  :collapsible: open

  * Click ``Particles`` (output of the last :bdg-secondary:`3D refinement` block) and select :bdg-primary:`Movie refinement`

  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

        - Set ``Last tilt for refinement`` to 4

        - Set ``Max resolution (A)`` to 3.2

        - Click on the **Frame refinement** tab

      .. md-tab-item:: Frame refinement

        - Specify the ``Input parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file `tomo-new-coarse-refinement-*_r01_02.bz2`

        - Set ``Spatial sigma`` to 200.0

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  * Click inside the :bdg-secondary:`Movie refinement` block to inspect the results:

    .. figure:: ../images/tutorial_tomo_movie_iter2.webp
      :alt: Iter 2

Step 15: Refinement after movie frame refinement
------------------------------------------------

.. nextpyp:: Additional refinement using new frame alignment parameters (:fa:`stopwatch` 1 hr)
  :collapsible: open

  * Click ``Frames`` (output of :bdg-secondary:`Movie refinement` block) and select :bdg-primary:`3D refinement (after movies)`

  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

        - Set ``Min number of tilts for refinement`` to 2

        - Set ``Max resolution (A)`` to 3.3

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Specify the ``Input parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file `tomo-flexible-refinement-*_r01_02.bz2`

        - Check ``Refine tilt-geometry``

        - Set ``Tilt-angle range (degrees)`` and ``Tilt-axis angle range (degrees)`` to 1.0

        - Set ``Tilt-axis angle range (degrees)`` to 10.0        

        - Check ``Refine particle alignments``

        - Set ``Particle rotation Phi range (degrees)``, ``Particle rotation Psi range (degrees)``, and ``Particle rotation Theta range (degrees)`` to 1.0

        - Set ``Particle translation range (A)`` to 10.0

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  * Click inside the :bdg-secondary:`3D refinement (after movies)` block to inspect the results:

    .. figure:: ../images/tutorial_tomo_after_movie_iter3.webp
      :alt: Iter 3

Step 16: Map sharpening
-----------------------

.. nextpyp:: Apply B-factor weighting in frequency space (:fa:`stopwatch` <1 min)
  :collapsible: open

  * Click ``Frames`` (output of :bdg-secondary:`3D refinement (after movies)` block) and select :bdg-primary:`Post-processing`

  * Go to the **Post-processing** tab:

    .. md-tab-set::

      .. md-tab-item:: Post-processing

        - Specify the ``First half map`` by clicking on the icon :fa:`search` and selecting the file `tomo-flexible-refinement-*_r01_half1.mrc` (output of the :bdg-secondary:`3D refinement (after movies)` block)

        - Set ``Threshold value`` to 0.35

        - Set ``B-factor method`` to adhoc

        - Set ``Adhoc value (A^2)`` to -25

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  * You can inspect the result by clicking inside the :bdg-secondary:`Post-processing` block:

    .. figure:: ../images/tutorial_tomo_final_map.webp
      :alt: Final map

.. info::

  Running times were measured running all tilt-series in parallel on nodes with 124 vCPUs, 720GB RAM, and 3TB of local SSDs
