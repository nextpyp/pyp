######################################
Classification tutorial (EMPIAR-10304)
######################################

This tutorial shows how to convert raw tilt-series from `EMPIAR-10304 (E. coli. ribosomes) <https://www.ebi.ac.uk/empiar/EMPIAR-10304/>`_ into a ~4.9A resolution structure and resolve 8 different conformations. 

Total running time required to complete this tutorial: 32 hrs.

We first use the command line to download and decompress a tbz file containing: 1) a script to download the raw tilt-series from EMPIAR, 2) corresponding metadata with tilt angles and acquisition order, and 3) an initial model:

.. code-block:: bash

  # cd to a location in the shared file system and run:

  wget https://nextpyp.app/files/data/nextpyp_class_tutorial.tbz
  tar xfz nextpyp_class_tutorial.tbz
  source download_10304.sh

.. note::

  Downloading the raw data from EMPIAR can take several minutes.

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

    * Select the new project from the **Dashboard** and click :badge:`Open,badge-primary`

    * The newly created project will be empty and a **Jobs** panel will appear on the right

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

        - Type ``*.mrc`` in the filter box (lower right) and click on the icon :fa:`filter,text-primary` to verify your selection. 12 matches should be displayed

        - Click :badge:`Choose File Pattern,badge-primary` to save your selection

        - Click on the **Microscope parameters** tab

      .. tabbed:: Microscope parameters

        - Set ``Pixel size (A)`` to 2.1

        - Set ``Acceleration voltage (kV)`` to 300

        - Set ``Tilt-axis angle (degrees)`` to 90.0

    * Click :badge:`Save,badge-primary` and the new block will appear on the project page

    * The block is in the modified state (indicated by the :fa:`asterisk` sign) and is ready to be executed

    * Clicking the button :badge:`Run,badge-primary` will show another dialog where you can select which blocks to run:

    * Since there is only one block available, simply click on :badge:`Start Run for 1 block,badge-primary`. This will launch a process that reads one tilt image, applies the gain reference (if applicable) and displays the resulting image inside the block

    * Click inside the block to see a larger version of the image


Step 3: Pre-processing
----------------------

.. dropdown:: Movie frame alignment, CTF estimation and tomogram reconstruction (:fa:`stopwatch` 4 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Tilt-series` (output of the :badge:`Tomography (from Raw Data),badge-secondary` block) and select :badge:`Pre-processing,badge-primary`

    * Go to the **Frame alignment** tab:

      .. tabbed:: Frame alignment

        - Check ``Single-file tilt-series``

        - Click on the **CTF determination** tab

      .. tabbed:: CTF determination

        - Set ``Max resolution`` to 5.0

        - Click on the **Tomogram reconstruction** tab

      .. tabbed:: Tomogram reconstruction

        - Check ``Erase fiducials``

        - Set ``Binning factor for reconstruction`` to 12

        - Set ``Thickness of reconstruction (unbinned voxels)`` to 3072

        - Uncheck ``Resize squares to closest multiple of 512``

        - Click on the **Resources** tab

      .. tabbed:: Resources

        - Set ``Threads per task`` to 42

        - Set ``Memory per task`` to 100

        - Set other runtime parameters as needed (see :doc:`Computing resources<../reference/computing>`)

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. Follow the status of the run in the **Jobs** panel


Step 4: Particle picking
--------------------------

.. dropdown:: Particle detection from tomograms (:fa:`stopwatch` 2 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

    * Go to the **Particle detection** tab:

      .. tabbed:: Particle detection

        - Set ``Detection radius (A)`` to 80

        - Set ``Detection method`` to size-based

        - Set ``Threshold for contamination detection`` to 2.0

        - Set ``Minimum contamination size (voxels)`` to 60

        - Set ``Minimum distance between particles`` to 2

        - Check ``Local refinement``

        - Set ``Z-axis detection range (binned voxels)`` to 40

        - Set ``Particle detection threshold`` to 2

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Navigate to the :badge:`Particle-Picking,badge-primary` block to inspect the coordinates

.. note::

    In this tutorial, we use the ``size-based`` method for particle detection. Other methods are available, including manual picking, spherical picking, and neural network-based picking.

.. seealso::

    * :doc:`2D particle picking<picking2d>`
    * :doc:`3D particle picking<picking3d>`
    * :doc:`Pattern mining<milopyp>`

Step 5: Reference-based refinement
----------------------------------

.. dropdown:: Reference-based particle alignment (:fa:`stopwatch` 26 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-secondary` block) and select :badge:`Particle refinement,badge-primary`

    * Go to the **Sample** tab:

      .. tabbed:: Sample

        - Set ``Molecular weight (kDa)`` to 2000

        - Set ``Particle radius (A)`` to 150

        - Click on the **Extraction** tab

      .. tabbed:: Extraction

        - Set ``Box size (pixels)`` to 64

        - Set ``Image binning`` to 4

        - Click on the **Refinement** tab

      .. tabbed:: Refinement

        - Specify the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary`, navigating to the folder where you downloaded the data for the tutorial, and selecting the file `EMPIAR-10304_init_ref.mrc`

        - Click :fa:`search,text-primary` in ``Alignments from sub-volume averaging`` to select the initial parameters text file ``tomo-preprocessing-*_original_volumes.txt`` from :badge:`Pre-processing,badge-secondary`

        - Set ``Max resolution (A)`` to 22.0

        - Check ``Use signed correlation``

        - Check ``Skip refinement``

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``First exposure for refinement`` to 15

        - Set ``Last exposure for refinement`` to 25

        - Set ``Optimizer - Exhaustive search points`` to 5000000

        - Check ``Refine particle alignments``

        - Set ``Phi range (degrees)``, ``Psi range (degrees)`` and ``Theta range (degrees)`` to 180

        - Set ``Translation range (voxels)`` to 50

        - Check ``Invert CTF handedness``

        - Click on the **Reconstruction** tab

      .. tabbed:: Reconstruction

        - Set ``Max tilt-angle`` to 50

        - Set ``Min tilt-angle`` to -50

        - Click on the **Resources** tab

      .. tabbed:: Resources

        - Set ``Threads per task`` to the maximum allowable by your system

        - Set ``Memory per task`` to at least 4x the number of ``Threads per task``

        - Set ``Walltime per task`` to 72:00:00

        - Set the ``Threads``, ``Memory``, and ``Walltime`` parameters for the ``Merge`` job to match the settings above

    * :badge:`Save,badge-primary` your changes, click :badge:`Run,badge-primary` and :badge:`Start Run for 1 block,badge-primary`

    * One round of refinement and reconstruction will be executed. Click inside the block to see the results

.. tip::


Step 6. Filter particles
------------------------

.. dropdown:: Identify duplicates and particles with low alignment scores (:fa:`stopwatch` 3 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of the :badge:`Particle refinement,badge-secondary` block) and select :badge:`Filter particles,badge-primary`

    * Go to the **Particle filtering** tab:

      .. tabbed:: Particle filtering

        - Set ``Score threshold`` to 15.0

        - Set ``Min distance between particles (A)`` to 20

        - Specify the location of ``Input parameter file`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_02.par.bz2`

        - Set ``Lowest tilt-angle`` to -7.0

        - Set ``Highest tilt-angle`` to 7.0

        - Set ``Min number of projections per particle`` to 1

        - Check ``Generate reconstruction after filtering``

        - Click on the **Refinement** tab

      .. tabbed:: Refinement

        - Specify the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_02.mrc`

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. You can see how many particles were left after filtering by looking at the job logs.


Step 7 (optional): Permanently remove bad particles
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


Step 8. Fully constrained refinement
------------------------------------

.. dropdown:: Tilt-geometry parameters and particle poses are refined in this step (:fa:`stopwatch` 10 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of the :badge:`Particle filtering,badge-secondary` block) and select :badge:`Particle refinement,badge-primary`

      .. tabbed:: Extraction

        - Set ``Box size (pixels)`` to 256

        - Set ``Image binning`` to 1

        - Click on the **Refinement** tab

      .. tabbed:: Refinement

        - Specify the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-fine-refinement-*_r01_02.mrc`

        - Select the location of the ``Initial parameter file`` by clicking on the icon :fa:`search,text-primary` and selecting the file `tomo-fine-refinement-*_r01_02.par.bz2` (select the file ``tomo-fine-refinement-*_r01_02_clean.par.bz2`` if bad particles were permanently removed in the previous step)

        - Set ``Max resolution (A)`` to 18:14

        - Check ``Use signed correlation``

        - Set ``Last iteration`` to 3

        - Check ``Skip refinement``

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``First exposure for refinement`` to 15

        - Set ``Last exposure for refinement`` to 25

        - Set ``Optimizer - Exhaustive search points`` to 0

        - Set ``Optimizer - Max step length`` to 100

        - Check ``Refine tilt-geometry``

        - Check ``Refine particle alignments``

        - Set ``Phi range``, ``Psi range`` and ``Theta range`` to 30.0

        - Set ``Translation range (voxels)`` to 30.0

        - Click on the **Exposure weighting** tab

      .. tabbed:: Exposure weighting

        - Check ``Dose weighting``

        - Check ``Global weights``

        - Set ``Frame weight fraction`` to 4

      .. tabbed:: Resources

        - Set ``Threads per task`` to the maximum allowable by your system

        - Set ``Memory per task`` to at least 6x the number of ``Threads per task``
        

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to execute three rounds of refinement and reconstruction

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results

Step 9: Create shape mask
--------------------------

.. dropdown:: Use most recent reconstruction to create a shape mask (:fa:`stopwatch` <1 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of :badge:`Particle refinement,badge-secondary` block) and select :badge:`Masking,badge-primary`

    * Go to the **Masking** tab:

      .. tabbed:: Masking

        - Select the ``Input map`` by click on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_03.mrc`

        - Set ``Threshold for binarization`` to 0.4

        - Check ``Use normalized threshold``

        - Set ``Width of cosine edge (pixels)`` to 8

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to run the job

    * Click on the menu icon :fa:`bars, text-primary` of the :badge:`Masking,badge-secondary` block, select the :badge:`Show Filesystem Location` option, and :badge:`Copy,badge-primary` the location of the block in the filesystem (we will use this in the next step))

    * Click inside the :badge:`Masking,badge-secondary` block to inspect the results of masking


Step 10. Region-based local refinement
--------------------------------------

.. dropdown:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 25 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Edit the settings of the existing :badge:`Particle refinement,badge-secondary` block and go to the **Refinement** tab:

    * Go to the **Refinement** tab:

      .. tabbed:: Refinement

        - Set ``Max resolution (A)`` to 18:14:12:10:8:6:5

        - Set ``Last iteration`` to 8

        - Specify the location of the ``Shape mask`` produced in Step 10 by clicking on the icon :fa:`search, text-primary`, navigating to the location of the :badge:`Masking,badge-secondary` block by copying the path we saved above, and selecting the file `frealign/maps/mask.mrc`

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``First exposure for refinement`` to 18

        - Set ``Last exposure for refinement`` to 22

        - Set ``Number of regions`` to 8,8,2

        - Set ``Tilt-angle range (degrees)`` and ``Tilt-axis range (degrees)`` to 5.0 

        - Set ``Phi range``, ``Psi range`` and ``Theta range`` to 5.0

        - Set ``Translation range (voxels)`` to 20.0

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to run the job

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results

Step 11: Particle-based CTF refinement
--------------------------------------

.. dropdown:: Per-particle CTF refinement using most recent reconstruction (:fa:`stopwatch` 2 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Edit the settings of the existing :badge:`Particle refinement,badge-secondary` block and go to the **Refinement** tab:

    * Go to the **Refinement** tab:

      .. tabbed:: Refinement

        - Set ``Max resolution (A)`` to 18:14:12:10:8:6:5:4.5

        - Set ``Last iteration`` to 9

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``First exposure for refinement`` to 15

        - Set ``Last exposure for refinement`` to 25

        - Uncheck ``Refine tilt-geometry``

        - Uncheck ``Refine particle alignments``

        - Check ``Refine CTF per-particle``

        - Set ``Defocus 1 range (A)`` and ``Defocus 2 range (A)`` to 2000.0

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results

Step 12: Region-based refinement after CTF refinement
-----------------------------------------------------

.. dropdown:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 20 min)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Edit the settings of the existing :badge:`Particle refinement,badge-secondary` block and go to the **Refinement** tab:

      .. tabbed:: Refinement

        - Set ``Max resolution (A)`` to 18:14:12:10:8:6:5:4.5:6:5:4.5

        - Set ``Last iteration`` to 12

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Set ``First exposure for refinement`` to 18

        - Set ``Last exposure for refinement`` to 22

        - Set ``Number of regions`` to 16,16,4

        - Set ``Optimizer - Max step length`` to 20.0

        - Check ``Refine tilt-geometry``

        - Set ``Translation range (pixels)`` to 20.0

        - Check ``Refine particle alignments``

        - Uncheck ``Refine CTF per-particle``

        - Click on the **Exposure weighting** tab

      .. tabbed:: Exposure weighting

        - Set ``Frame weight fraction`` to 2

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to run the job

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results

Step 13: 3D classification
--------------------------

.. dropdown:: Constrained classification (:fa:`stopwatch` 3 hr)
    :container: + shadow
    :title: bg-primary text-white text-left
    :open:

    * Click on :guilabel:`Particles` (output of the :badge:`Particle refinement,badge-secondary` block) and select :badge:`Particle refinement,badge-primary` to create a new block

    * Go to the **Refinement** tab:

      .. tabbed:: Refinement

        - Specify the location of the ``Initial model`` by clicking on the icon :fa:`search, text-primary` and selecting the file `tomo-coarse-refinement-*_r01_12.mrc`

        - Select the location of the ``Initial parameter file`` by clicking on the icon :fa:`search,text-primary` and selecting the file `tomo-coarse-refinement-*_r01_12.par.bz2`

        - Set ``Max resolution (A)`` to 8

        - Set ``Last iteration`` to 20

        - Click on the **Constrained refinement** tab

      .. tabbed:: Constrained refinement

        - Uncheck ``Refine tilt-geometry``

        - Uncheck ``Refine particle alignments``

        - Click on the **Classification** tab

      .. tabbed:: Classification

        - Set ``Number of classes`` to 8

        - Click on the **Exposure weighting** tab

      .. tabbed:: Exposure weighting

        - Specify the location of the ``External weights`` by clicking on the icon :fa:`search, text-primary` and selecting the file `frealign/weights/global_weights.txt` from the file location of the previous block

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

    * Click inside the :badge:`Particle refinement,badge-secondary` block to inspect the results

.. tip::

    Click on the round blue markers (top right of the page) to inspect different classes or go to the **Class view** or **Classes Movie** tabs to show all classes simultaneously

.. note::

  Running times were measured running all tilt-series in parallel on nodes with 124 vCPUs, 720GB RAM, and 3TB of local SSDs

.. seealso::

    * :doc:`Single-particle tutorial<spa_empiar_10025>`
    * :doc:`Single-particle (on-the-fly)<stream_spr>`
    * :doc:`Tomography tutorial<tomo_empiar_10164>`
    * :doc:`Tomography (on-the-fly)<stream_tomo>`
