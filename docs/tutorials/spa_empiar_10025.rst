#######################################
Single-particle tutorial (EMPIAR-10025)
#######################################

This tutorial shows how to convert raw movies from `EMPIAR-10025 (T20S proteasome) <https://www.ebi.ac.uk/empiar/EMPIAR-10025/>`_ into a ~3A resolution structure.

Total running time required to complete this tutorial: 45m.

We first use the command line to download and decompress a tbz file containing a subset of 20 movies, the gain reference, and an initial model:

.. code-block:: bash

  # cd to a location in the shared file system and run:

  wget https://nextpyp.app/files/data/nextpyp_spr_tutorial.tbz
  tar xvfz nextpyp_spr_tutorial.tbz

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

    .. figure:: ../images/tutorial_spa_new.webp
      :alt: Create new project

  * Select the new project from the **Dashboard** and click :badge:`Open,badge-primary`

    .. figure:: ../images/tutorial_spa_open.webp
      :alt: Select new project

  * The newly created project will be empty and a **Jobs** panel will appear on the right

    .. figure:: ../images/tutorial_spa_empty.webp
      :alt: Empty project

Step 2: Import raw movies
-------------------------

.. dropdown:: Import the raw movies downloaded above (:fa:`stopwatch` <1 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click :badge:`Import Data,badge-primary` and select :badge:`Single Particle (from Raw Data),badge-primary`

    .. figure:: ../images/tutorial_spa_import_dialog.webp
      :alt: Import dialog

  * A form to enter parameters will appear:

    .. figure:: ../images/tutorial_spa_import_data.webp
      :alt: File browser

  * Go to the **Raw data** tab:

    .. tabbed:: Raw data

      - Set the ``Location`` of the raw data clicking on the icon :fa:`search,text-primary` and browsing to the directory where the you downloaded the raw data

      - Type ``14*.tif`` in the filter box (lower right) and click on the icon :fa:`filter,text-primary` to verify your selection. 20 matches should be displayed

      - Click :badge:`Choose File Pattern,badge-primary`

      - Click on the **Gain reference** tab

      .. figure:: ../images/tutorial_spa_import_browser.webp
        :alt: File browser

    .. tabbed:: Gain reference

      - Set the ``Location`` of the gain reference by clicking the icon :fa:`search,text-primary` and navigating to the directory where you downloaded the data for the tutorial. Select the file ``Gain.mrc`` and click :badge:`Choose File,badge-primary`

      - Check ``Flip vertically``

      - Click on the **Microscope parameters** tab

      .. figure:: ../images/tutorial_spa_import_gain.webp
        :alt: File browser

    .. tabbed:: Microscope parameters

      - Set ``Pixel size (A)`` to 0.66

      - Set ``Acceleration voltage (kV)`` to 300

      .. figure:: ../images/tutorial_spa_import_scope.webp
        :alt: Project dashboard

  * Click :badge:`Save,badge-primary` and the new block will appear on the project page

    .. figure:: ../images/tutorial_spa_import_modified.webp
      :alt: Project dashboard

  * The block is in the modified state (indicated by the :fa:`asterisk` sign, top bar) and is ready to be executed

  * Clicking the button :badge:`Run,badge-primary` will show another dialog where you can select which blocks to run. Since there is only block available, simply click on :badge:`Start Run for 1 block,badge-primary`. This will launch a process that reads the first movie, applies the gain reference and displays a thumbnail inside the :badge:`Single Particle (from Raw Data),badge-secondary` block

    .. figure:: ../images/tutorial_spa_import_done.webp
      :alt: Gain thumbnail

.. tip::

    Click inside the :badge:`Single Particle (from Raw Data),badge-secondary` block to see a larger version of the image

Step 3: Pre-processing
----------------------

.. dropdown:: Movie frame alignment, CTF estimation and particle picking (:fa:`stopwatch` 2 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click on :guilabel:`Movies` (output of :badge:`Single Particle (from Raw Data),badge-secondary` block) and select :badge:`Pre-processing,badge-primary`

    .. figure:: ../images/tutorial_spa_pre_process_dialog.webp
      :alt: File browser

  * Go to the **Particle detection** tab:

    .. tabbed:: Particle detection

      * Set ``Particle radius (A)`` to 65

      * Set ``Detection method`` to all

      * Set ``Min distance (pixels)`` to 40

      * Click on the **Resources** tab

    .. tabbed:: Resources

      * Set ``Threads per task`` to 7

      * Set ``Memory per task`` to 14

      * Set other runtime parameters as needed (see :doc:`Computing resources<../reference/computing>`)

  * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. You can monitor the status of the run using the **Jobs** panel

    .. figure:: ../images/tutorial_spa_pre_process_modified.webp
      :alt: File browser

  * Click inside the :badge:`Pre-processing,badge-secondary` block to inspect the results (you don't need to wait until processing is done to do this). Results will be grouped into tabs:

    .. tabbed:: Plots

      .. figure:: ../images/tutorial_spa_pre_process_page.webp
        :alt: Dataset statistics

    .. tabbed:: Table

      .. figure:: ../images/tutorial_spa_pre_process_table.webp
        :alt: Table view

    .. tabbed:: Gallery

      .. figure:: ../images/tutorial_spa_pre_process_gallery.webp
        :alt: Gallery view

    .. tabbed:: Micrograph

      Data processing details (particle picking, drift trajectory, CTF profile, power spectrum)

      .. figure:: ../images/tutorial_spa_pre_process_micrographs.webp
        :alt: Micrograph view

.. tip::

  While on the **Micrographs** tab, use the navigation bar at the top of the page to look at the results for other micrographs

Step 4: Reference-based refinement
----------------------------------

.. dropdown:: Reference-based particle alignment (:fa:`stopwatch` 3 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click on :guilabel:`Particles` (output of :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle refinement,badge-primary`

    .. figure:: ../images/tutorial_spa_coarse_dialog.webp
      :alt: File browser

  * Go to the **Sample** tab:

    .. tabbed:: Sample

      - Set ``Molecular weight (kDa)`` to 700

      - Set ``Particle radius (A)`` to 80

      - Set ``Symmetry`` to D7

      - Click on the **Extraction** tab

      .. figure:: ../images/tutorial_spa_coarse_sample.webp
        :alt: File browser

    .. tabbed:: Extraction

      - Set ``Box size (pixels)`` to 128

      - Set ``Image binning`` to 4

      - Click on the **Refinement** tab

      .. figure:: ../images/tutorial_spa_coarse_extract.webp
        :alt: File browser

    .. tabbed:: Refinement

      - Set the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary`, navigating to the folder where you downloaded the data for the tutorial, selecting the file  `EMPIAR-10025_init_ref.mrc`, and clicking :badge:`Choose File,badge-primary`

      - Set ``Last iteration`` to 5

      - Set ``Max resolution (A)`` to 8:7:6

      - Check ``Use signed correlation``

      - Click on the **Reconstruction** tab

      .. figure:: ../images/tutorial_spa_coarse_refinement.webp
        :alt: File browser

    .. tabbed:: Reconstruction

      - Set ``Fraction of particles`` to 0

      .. figure:: ../images/tutorial_spa_coarse_reconstruction.webp
        :alt: File browser

  * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    The new block will appear on the **Dashboard** and a thumbnail will be displayed inside after the run is finished

    .. figure:: ../images/tutorial_spa_coarse_modified.webp
      :alt: File browser

    This process executes four rounds of global orientation search (iterations 2-5). The fraction of good particles at each iteration will be determined automatically (``Fraction of particles`` = 0) and used for reconstruction

  * Click inside the :badge:`Pre-processing,badge-secondary` block to inspect the results:

    .. figure:: ../images/tutorial_spa_coarse_iter5.webp
      :alt: Iteration 5

Step 5: Filter bad particles
----------------------------

.. dropdown:: Identify particles with low alignment scores (:fa:`stopwatch` 1 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click on :guilabel:`Particles` (output of :badge:`Particle refinement,badge-secondary` block) and select :badge:`Particle filtering,badge-primary`

    .. figure:: ../images/tutorial_spa_fine_dialog.webp
      :alt: File browser

  * Go to the **Particle filtering** tab:

    .. tabbed:: Particle filtering

      - Check ``Automatic score threshold``

      - Set ``Min distance between particles (A)`` to 20

      - Select the ``Input parameter file`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-coarse-refinement-*_r01_05.par.bz2`

      - Check ``Generate reconstruction after filtering``

      - Click on the **Refinement** tab

    .. tabbed:: Refinement

      - Select the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-coarse-refinement-*_r01_05.mrc`

  * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to execute particle cleaning and produce a reconstruction with only the clean particles

  * Click inside the :badge:`Filter particles,badge-secondary` block to look at the reconstruction after cleaning:

    .. figure:: ../images/tutorial_spa_cleaning_iter2.webp
      :alt: Iteration 2

Step 6 Permanently remove bad particles
---------------------------------------

.. dropdown:: Permanently remove bad particles to improve efficiency of steps downstream (:fa:`stopwatch` <1 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Edit the settings of the :badge:`Particle filtering,badge-secondary` block by clicking on the icon :fa:`bars, text-primary` and selecting the :fa:`edit, text-primary` Edit option

  * Go to the **Particle filtering** tab

    .. tabbed:: Particle filtering

      - Check ``Permanently remove particles``

      - Uncheck ``Generate reconstruction after filtering``

  * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to launch the job

Step 7: Particle refinement
---------------------------

.. dropdown:: Reconstruction and additional refinement using 2x binned particles (:fa:`stopwatch` 9 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click on :guilabel:`Particles` (output of :badge:`Filter particles,badge-secondary` block) and select :badge:`Particle refinement,badge-primary`

    .. figure:: ../images/tutorial_spa_fine_dialog.webp
      :alt: File browser

  * Go to the **Extraction** tab:

    .. tabbed:: Extraction

      - Set ``Box size (pixels)`` to 256

      - Set ``Image binning`` to 2

      - Click on the **Refinement** tab

    .. tabbed:: Refinement

      - Select the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-fine-refinement-*_r01_02.mrc`

      - Select the ``Input parameter file`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-fine-refinement-*_r01_02_clean.par.bz2`

      - Set ``Last iteration`` to 6

      - Set ``Search mode`` to local

      - Set ``Max resolution (A)`` to 6:4:3

      - Check ``Use signed correlation``

  * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to launch the job

  * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results:

    .. figure:: ../images/tutorial_spa_fine_iter2.webp
      :alt: Iteration 2

.. tip::

  Use the navigation bar at the top left of the page to look at the results for different iterations

Step 8: Create shape mask
-------------------------

.. dropdown:: Use most recent reconstruction to build a shape mask (:fa:`stopwatch` <1 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click on :guilabel:`Particles` (output of :badge:`Particle refinement,badge-secondary` block) and select :badge:`Masking,badge-primary`

  * Enter parameter values for the **Masking** tab:

    .. tabbed:: Masking

      - Select the ``Input map`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-coarse-refinement-*_r01_06.mrc`

      - Set ``Threshold for binarization`` to 0.3

  * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to launch the job

  * Click on the icon :fa:`bars, text-primary` of the :badge:`Masking,badge-secondary` block, select the :badge:`Show Filesystem Location` option, and :badge:`Copy,badge-primary` the location of the block in the filesystem (we will use this in the next step))

  * Click inside the :badge:`Masking,badge-secondary` block to inspect the results of masking

Step 9: Local refinement
------------------------

.. dropdown:: Additional refinement iterations using 2x binned data (:fa:`stopwatch` 2 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Go one block upstream to the :badge:`Particle refinement,badge-secondary` block, click on the icon :fa:`bars, text-primary` and select the :fa:`edit, text-primary` Edit option from the menu 

  * Go to the **Refinement** tab:

    .. tabbed:: Refinement

      - Set ``Last iteration`` to 7

      - Select the ``Shape mask`` by clicking on the icon :fa:`search, text-primary`, navigating to the path of the :badge:`Masking,badge-secondary` block copied above, and selecting the file `frealign/maps/mask.mrc`

  * Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary`. We now need to uncheck the box for the :badge:`Masking,badge-secondary` block (since we don't want to re-run this block), then click :badge:`Start Run for 1 block,badge-primary`

  * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results:

    .. figure:: ../images/tutorial_spa_fine_iter7.webp
      :alt: Iteration 7

Step 10: Particle-based CTF refinement
--------------------------------------

.. dropdown:: Per-particle CTF refinement using most recent reconstruction (:fa:`stopwatch` 9 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on the menu icon :fa:`bars, text-primary` from the :badge:`Particle refinement,badge-secondary` block and choose the :fa:`edit, text-primary` Edit option.

    * Go to the **Refinement** tab:

      .. tabbed:: Refinement

        - Set ``Last iteration`` to 8

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``Number of regions`` to 8,8

        - Check ``Refine CTF per-particle``

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results


Step 11: Movie frame refinement
-------------------------------

.. dropdown:: Particle-based movie-frame alignment and data-driven exposure weighting (:fa:`stopwatch` 8 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click :guilabel:`Particle set` (output of :badge:`Particle refinement,badge-secondary` block) and select :badge:`Movie refinement,badge-primary`

  * Go to the **Refinement** tab:

    .. tabbed:: Refinement

      - Select the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-coarse-refinement-*_r01_07.mrc`

      - Select the ``Input parameter`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-coarse-refinement-*_r01_07.par.bz2`

      - Set ``Last iteration`` to 3

      - Check ``Skip refinement``

      - Set ``Max resolution (A)`` to 3

      - Go to the **Constrained refinement** tab

    .. tabbed:: Constrained refinement

      - Set ``Last exposure for refinement`` to 60

      - Check ``Movie frame refinement``

      - Check ``Show advanced options``

      - Check ``Regularize translations``

      - Set ``Spatial sigma`` to 15

      - Go to the **Exposure weighting** tab

    .. tabbed:: Exposure weighting

      - Check ``Dose weighting``

  * Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to launch Movie refinement. Uncheck the box for the :badge:`Masking,badge-secondary` block and click :badge:`Start Run for 1 block,badge-primary`

  * Click inside the :badge:`Movie refinement,badge-secondary` block to inspect the results:

    .. figure:: ../images/tutorial_spa_movie_iter3.webp
      :alt: Iteration 3

Step 12: Refinement after movie frame refinement
------------------------------------------------

.. dropdown:: Additional refinement using new frame alignment parameters (:fa:`stopwatch` 8 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click on the menu icon :fa:`bars, text-primary` from the :badge:`Movie refinement,badge-secondary` block and choose the :fa:`edit, text-primary` Edit option.

  * Go to the **Refinement** tab:

    .. tabbed:: Refinement

      - Set ``Last iteration`` to 4

      - Uncheck ``Skip refinement``

      - Click on the **Constrained refinement** tab

    .. tabbed:: Constrained refinement

      - Uncheck ``Movie frame refinement``

  * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

  * Click inside the :badge:`Movie refinement,badge-secondary` block to inspect the results:

    .. figure:: ../images/tutorial_spa_final_map.webp
      :alt: Final map

Step 13: Map sharpening
-----------------------

.. dropdown:: Apply B-bactor weighting in frequency space (:fa:`stopwatch` <1 min)
  :container: + shadow
  :title: bg-primary text-white text-left
  :open:

  * Click :guilabel:`Frames` (output of :badge:`Movie refinement,badge-secondary` block) and select :badge:`Post-processing,badge-primary`

  * Go to the **Post-processing** tab:

    .. tabbed:: Post-processing

      - Select the ``First half map`` by clicking on the icon :fa:`search, text-primary` and selecting the file `sp-flexible-refinement-*_r01_half1.mrc`

      - Set ``Automask threshold`` to 0.5

      - Set ``Adhoc B-factor (A^2)`` to -50

  * Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary`. Uncheck the box for the :badge:`Masking,badge-secondary` block and click :badge:`Start Run for 1 block,badge-primary`

  * Click inside the :badge:`Map sharpening,badge-secondary` block to inspect the results:

    .. figure:: ../images/tutorial_spa_post_processing.webp
      :alt: Post processing

.. note::

  Running times were measured running micrographs in parallel on nodes with 124 vCPUs, 720GB RAM, and 3TB of local SSDs

.. seealso::

    * :doc:`Single-particle session<stream_spr>`
    * :doc:`Tomography tutorial<tomo_empiar_10164>`
    * :doc:`Classification tutorial<tomo_empiar_10304>`
    * :doc:`Tomography session<stream_tomo>`
