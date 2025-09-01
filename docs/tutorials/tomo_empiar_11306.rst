########################################
In-situ ribosome tutorial (EMPIAR-11306)
########################################

This tutorial shows how to convert raw tilt-series from `EMPIAR-11306 (plasma FIB lamellae of HeLa cells) <https://www.ebi.ac.uk/empiar/EMPIAR-11306/>`_ into a ~5A resolution structure of 80S ribosomes.

.. admonition::

  * Total running time required to complete this tutorial: **?? hrs**.

We first download the raw data from the EMPIAR database and an initial model from the EMDB:

.. code-block:: bash

  # cd to a location in the shared file system and download the raw data from EMPIAR:

  mkdir EMPIAR-11306
  wget -m -q -nd -P ./EMPIAR-11306 ftp://ftp.ebi.ac.uk/empiar/world_availability/11306  
  
  # download the initial model from the EMDB
  wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-15636/map/emd_15636.map.gz

.. note::

  Depending on the speed of your network connection, downloading the raw data from EMPIAR can take several hours.

Open your browser and navigate to the url of your ``nextPYP`` instance (e.g., ``https://nextpyp.myorganization.org``).

Step 1: Create a new project
----------------------------

.. nextpyp::We will create a new project for this tutorial
  :collapsible: open

  * Click on :bdg-primary:`Create new project`, give the project a name, and select :bdg-primary:`Create`

  * Select the new project from the **Dashboard** and click :bdg-primary:`Open`

Step 2: Import raw tilt-series
------------------------------

.. nextpyp:: Import the raw tilt-series downloaded above (:fa:`stopwatch` <1 min)
  :collapsible: open

  * Go to :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`

  * Go to the **Raw data** tab:

    .. md-tab-set::
      :class: custom-tab-set-style

      .. md-tab-item:: Raw data

        - Set the ``Location`` of the raw data by clicking on the icon :fa:`search` and browsing to the directory where the you downloaded the raw movie frames

        - Type ``*.eer`` in the filter box (lower right) and click on the icon :fa:`filter` to verify your selection. 6300 matches should be displayed

        - Click :bdg-primary:`Choose File Pattern` to save your selection

        - Set the ``Location of mdoc files`` by clicking on the icon :fa:`search` and browsing to the directory where the you downloaded the raw data

        - Type ``*.mdoc`` in the filter box and click on the icon :fa:`filter` to verify your selection. 180 matches should be displayed

        - Click :bdg-primary:`Choose File Pattern` to save your selection

        - Click on the **Microscope parameters** tab

      .. md-tab-item:: Gain reference

        - Set the ``Location`` by clicking on the icon :fa:`search` and browsing to the location of the gain reference file ``gain_ref/20220406_175020_EER_GainReference.gain``

        - Click :bdg-primary:`Choose File` to save your selection

      .. md-tab-item:: Microscope parameters

        - Set ``Pixel size (A)`` to 1.9

        - Set ``Acceleration voltage (kV)`` to 300

        - Set ``Tilt-axis angle (degrees)`` to 90

  * Click :bdg-primary:`Save` and the new block will appear on the project page

  * Clicking the button :bdg-primary:`Run`, then :bdg-primary:`Start Run for 1 block`. This will launch a process that reads one tilt image, applies the gain reference and displays the resulting image inside the block

  * Click inside the block to see a larger version of the image

Step 3: Pre-processing
----------------------

.. nextpyp:: :fa:`stopwatch` 4 min - Movie frame alignment, CTF estimation and tomogram reconstruction
  :collapsible: open

  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

  * Go to the **Frame alignment** tab:

    .. md-tab-set::

      .. md-tab-item:: Frame alignment

        - Set ``Frame pattern`` to *TILTSERIES_SCANORD_ANGLE_EER.eer*

        - Set ``Alignment method`` to *motioncor3 (GPU needed)*

        - Set ``EER frames to average`` to 10

        - Click on the **CTF determination** tab

      .. md-tab-item:: CTF determination

        - Set ``Max resolution`` to 7.0

        - Set ``Min defocus (A)`` to 25000

        - Click on the **Tilt-series alignment** tab

      .. md-tab-item:: Tilt-series alignment

        - Uncheck ``Reshape tilt-images into squares``

        - Set ``Alignment method`` to *aretomo2 (GPU required)*

        - Click on the **Tomogram reconstruction** tab

      .. md-tab-item:: Tomogram reconstruction

        - Set ``Thickness of reconstruction (unbinned voxels)`` to 1535

        - Set ``Binning factor for reconstruction`` to 12

        - Set ``Reconstruction method`` to *aretomo2 (GPU required)*

        - Check ``Use SART reconstruction``

        - Click on the **Resources** tab

      .. md-tab-item:: Resources

        - Set ``Split, Threads`` to 11

        - Set other runtime parameters as needed (see :doc:`Computing resources<../reference/computing>`)

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

Step 4: Particle picking
------------------------

.. nextpyp:: Particle detection from tomograms (:fa:`stopwatch` 2 min)
  :collapsible: open

  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle picking`

  * Go to the **Particle detection** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle detection

        - Set ``Detection method`` to *size-based*

        - Set ``Particle radius (A)`` to 100

        - Set ``Minimum distance between particles`` to 1

        - Check ``Local refinement``

        - Set ``Z-axis detection range (binned slices)`` to 80

        - Set ``Particle detection threshold`` to 1.1

        - Check ``Filter positions by relative contrast``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  * Navigate to the :bdg-primary:`Particle picking` block to inspect the coordinates

Step 5: Reference-based refinement
----------------------------------

.. nextpyp:: Reference-based particle alignment (:fa:`stopwatch` 14 hr)
    :collapsible: open

    * Click on ``Particles`` (output of the :bdg-secondary:`Particle picking` block) and select :bdg-primary:`Reference-based refinement`

    * Go to the **Sample** tab:

      .. md-tab-set::
        
        .. md-tab-item:: Sample

          - Set ``Molecular weight (kDa)`` to 3000

          - Set ``Particle radius (A)`` to 150

          - Click on the **Particle extraction** tab

        .. md-tab-item:: Particle extraction

          - Set ``Box size (pixels)`` to 64

          - Set ``Image binning`` to 4

          - Uncheck ``Skip gold fiducials``

          - Check ``Invert CTF handedness``

          - Click on the **Particle scoring function** tab

        .. md-tab-item:: Particle scoring function

          - Set ``Last tilt for refinement`` to 16

          - Set ``Max resolution (A)`` to 20.0

          - Click on the **Reference-based refinement** tab

        .. md-tab-item:: Reference-based refinement

          - Specify the location of the ``Initial model (*.mrc)`` by clicking on the icon :fa:`search`, navigating to the folder where you downloaded the data for the tutorial, and selecting the file `EMPIAR-10304_init_ref.mrc`

          - Set ``Particle rotation Phi range (degrees)`` and ``Particle rotation Psi range (degrees)`` to 360

          - Set ``Particle rotation Theta range (degrees)``` to 360

          - Set ``Rotation step (degrees)`` to 8

          - Set ``Particle translation range (A)`` to 50

          - Set ``Particle translation step (A)`` to 6

          - Click on the **Reconstruction** tab

        .. md-tab-item:: Reconstruction

          - Check ``Show advanced options``

          - Set ``Max tilt-angle (degrees)`` to 50

          - Set ``Min tilt-angle (degrees)`` to -50

          - Click on the **Resources** tab

        .. md-tab-item:: Resources

          - Set ``Split, Threads`` to the maximum allowable by your system

    * :bdg-primary:`Save` your changes, click :bdg-primary:`Run` and :bdg-primary:`Start Run for 1 block`

    * One round of refinement and reconstruction will be executed. Click inside the block to see the results


Step 6. Filter particles
------------------------

.. nextpyp:: Identify duplicates and particles with low alignment scores (:fa:`stopwatch` 3 min)
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Reference-based refinement` block) and select :bdg-primary:`Particle filtering`

  * Go to the **Particle filtering** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle filtering

          - Specify the location of ``Input parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file `tomo-reference-refinement-*_r01_02.bz2`

          - Set ``Score threshold`` to 7

          - Set ``Min distance between particles (unbinned pixels/voxels)`` to 100

          - Set ``Lowest tilt-angle (degrees)`` to -16

          - Set ``Highest tilt-angle (degrees)`` to 16

          - Check ``Generate reconstruction after filtering``

          - Check ``Permanently remove particles``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. You can see how many particles were left after filtering by looking at the job logs.

## TODO ##

Step 7. Fully constrained refinement
------------------------------------

.. nextpyp:: Tilt-geometry parameters and particle poses are refined in this step (:fa:`stopwatch` 10 min)
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle filtering` block) and select :bdg-primary:`3D refinement`

  * Go to the **Particle extraction** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle extraction

        - Set ``Box size (pixels)`` to 256

        - Set ``Image binning`` to 1

        - Click on the **Particle scoring function** tab

      .. md-tab-item:: Particle scoring function

        - Set ``Max resolution (A)`` to 18:14

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Select the location of the ``Initial parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file ``tomo-fine-refinement-*_r01_clean.bz2``

        - Set ``Last iteration`` to 3

        - Check ``Refine tilt-geometry``

        - Check ``Refine particle alignments``

        - Set ``Particle translation range (A)`` to 30.0

        - Click on the **Reconstruction** tab

      .. md-tab-item:: Reconstruction

        - Check ``Apply dose weighting``

        - Check ``Global weights``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to execute three rounds of refinement and reconstruction

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results

Step 8: Create shape mask
-------------------------

.. nextpyp:: Use most recent reconstruction to create a shape mask (:fa:`stopwatch` <1 min)
  :collapsible: open

  * Click on ``Particles`` (output of :bdg-secondary:`3D refinement` block) and select :bdg-primary:`Masking`

  * Go to the **Masking** tab:

    .. md-tab-set::

      .. md-tab-item:: Masking

        - Select the ``Input map (*.mrc)`` by click on the icon :fa:`search` and selecting the file `tomo-new-coarse-refinement-*_r01_03.mrc`

        - Set ``Threshold for binarization`` to 0.4

        - Set ``Width of cosine edge (pixels)`` to 8

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to run the job

  * Click on the menu icon :fa:`bars` of the :bdg-secondary:`Masking` block, select the :bdg-secondary:`Show Filesystem Location` option, and :bdg-primary:`Copy` the location of the block in the filesystem (we will use this in the next step))

  * Click inside the :bdg-secondary:`Masking` block to inspect the results of masking


Step 9. Region-based local refinement
--------------------------------------

.. nextpyp:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 25 min)
  :collapsible: open

  * Click on ``Particles`` (output of :bdg-secondary:`3D refinement` block) and select :bdg-primary:`3D refinement`

  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

        - Set ``First tilt for refinement`` to 18

        - Set ``Last tilt for refinement`` to 22

        - Set ``Max resolution (A)`` to 18:14:12:10:8:6:5
        
        - Set ``Masking strategy`` to *from file*

        - Specify the location of the ``Shape mask (*.mrc)`` produced in Step 9 by clicking on the icon :fa:`search`, navigating to the location of the :bdg-secondary:`Masking` block by copying the path we saved above, and selecting the file `frealign/maps/mask.mrc`

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Select the location of the ``Initial parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file ``tomo-new-coarse-refinement-*_r01_03.bz2``

        - Set ``Last iteration`` to 5

        - Set ``Number of regions`` to 8,8,2

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to run the job

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results

Step 10: Particle-based CTF refinement
--------------------------------------

.. nextpyp:: Per-particle CTF refinement using most recent reconstruction (:fa:`stopwatch` 2 hr)
  :collapsible: open

  * Click on ``Particles`` (output of :bdg-secondary:`3D refinement` block) and select :bdg-primary:`3D refinement`

  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

          - Set ``First tilt for refinement`` to 15

          - Set ``Last tilt for refinement`` to 25

          - Set ``Max resolution (A)`` to 4.5

          - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

          - Select the location of the ``Initial parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file ``tomo-new-coarse-refinement-*_r01_05.bz2``

          - Uncheck ``Refine tilt-geometry``

          - Uncheck ``Refine particle alignments``

          - Check ``Refine CTF per-particle``

          - Set ``Defocus 1 range (A)`` and ``Defocus 2 range (A)`` to 2000.0

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results

Step 11: Region-based refinement after CTF refinement
-----------------------------------------------------

.. nextpyp:: Constraints of the tilt-geometry are applied over local regions (:fa:`stopwatch` 20 min)
  :collapsible: open

  * Click on ``Particles`` (output of :bdg-secondary:`3D refinement` block) and select :bdg-primary:`3D refinement`

  * Go to the **Particle scoring function** tab:

    .. md-tab-set::

      .. md-tab-item:: Particle scoring function

        - Set ``First tilt for refinement`` to 18

        - Set ``Last tilt for refinement`` to 22

        - Set ``Max resolution (A)`` to 18:14:12:10:8:6:5:4.5:6:5:4.5

        - Click on the **Refinement** tab

      .. md-tab-item:: Refinement

        - Select the location of the ``Initial parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file ``tomo-new-coarse-refinement-*_r01_02.bz2``

        - Set ``Last iteration`` to 12

        - Check ``Refine tilt-geometry``

        - Check ``Refine particle alignments``

        - Uncheck ``Refine CTF per-particle``

        - Set ``Number of regions`` to 16,16,4

        - Set ``Optimizer - Max step length`` to 20.0

        - Click on the **Reconstruction** tab

      .. md-tab-item:: Reconstruction

        - Set ``Frame weight fraction`` to 2

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to run the job

  * Click inside the :bdg-secondary:`3D refinement` block to inspect the results

Step 12: 3D classification
--------------------------

.. nextpyp:: Constrained classification (:fa:`stopwatch` 3 hr)
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`3D classification`

  * Go to the **Classification** tab:

    .. md-tab-set::

      .. md-tab-item:: Classification

        - Select the location of the ``Initial parameter file (*.bz2)`` by clicking on the icon :fa:`search` and selecting the file `tomo-new-coarse-refinement-*_r01_02.bz2`

        - Set ``Last iteration`` to 20

        - Set ``Number of classes`` to 8

        - Uncheck ``Refine particle alignments``

        - Click on the **Reconstruction** tab

      .. md-tab-item:: Reconstruction

        - Specify the location of the ``External weights`` by clicking on the icon :fa:`search` and selecting the file ``frealign/weights/global_weight.txt`` from the file location of the previous block

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  * Click inside the :bdg-secondary:`3D classification` block to inspect the results

    .. tip::

        Click on the round blue markers (top right of the page) to inspect different classes or go to the **Class view** or **Classes Movie** tabs to show all classes simultaneously

.. info::

  Running times were measured running all tilt-series in parallel on nodes with 124 vCPUs, 720GB RAM, and 3TB of local SSDs
