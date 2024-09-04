=====================
Tomogram segmentation
=====================

``nextPYP`` can segment closed surfaces using an in-house algorithm, or open surfaces using `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_.

**Closed** surfaces are used to segment virions or vescicles, while **open** surfaces are used to segment irregular membranes.

Closed surfaces
---------------

1. Find centers of closed surfaces

    * Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle Picking,badge-primary`

    * Select the desired particle picking algorithm and corresponding parameters, see :doc:`Particle picking<picking>`

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. Follow the status of the run in the **Jobs** panel

    * Navigate to the :badge:`Reconstruction,badge-primary` tab to inspect the results of the detection

2. Segment individual surfaces

    * Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-secondary` block) and select :badge:`Segmentation (closed surfaces),badge-primary`

    * Set the desired `Segmentation radius tolerance`

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. Follow the status of the run in the **Jobs** panel

    * Navigate to the :badge:`Segmentation (closed surfaces),badge-primary` block and select the **Segmentation** tab to inspect the denoised tomograms

    * (optional) ``nextPYP`` calculates an implicit representation of the surface that only requires specifying one of several threshold values to detect the virion membrane. In many cases, the default value for the threshold gives reasonable results, but users can also manually select different thresholds using the web-based GUI. Select a virion from the table to show its 3D slices and the segmentation thresholds (8 different thresholds are shown as yellow contours in columns 1-8). The highlighted column number represents the current threshold selection (default is 1, click on a different column to select a better threshold). If none of the columns look reasonable (or if you want to ignore the current virion), select the last column labeled as "-". Repeat this process for all virion in the tilt-series and all tilt-series in the dataset

Open surfaces (membrain-seg)
----------------------------

    * Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Segmentation (open surfaces),badge-primary`

    * Select the desired parameters for MemBrain-Seg including the location of a pre-trained model

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. Follow the status of the run in the **Jobs** panel

    * Navigate to the :badge:`Segmentation (open surfaces),badge-primary` block to inspect the results of the segmentation

.. note::

    As of now, the membrain-seg results are only used for visual inspection or for processing outside of ``nextPYP``

.. seealso::

    * :doc:`Denoising<denoising>`
    * :doc:`Particle picking<picking>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`