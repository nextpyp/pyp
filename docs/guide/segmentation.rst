=====================
Tomogram segmentation
=====================

Closed surfaces
---------------

``nextPYP`` segments closed surfaces (such as virions or spherical vescicles) using an energy-based algorithm. Segmentation proceeds in two steps:

1. Find centers
~~~~~~~~~~~~~~~

The first step is to locate a marker inside each virion or vescicle using traditional particle picking.

* Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle Picking,badge-primary`

* Select the desired particle picking algorithm and corresponding parameters, see :doc:`Particle picking<picking>`

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Reconstruction,badge-primary` tab to inspect the results of the detection

2. Segment surfaces
~~~~~~~~~~~~~~~~~~~

Once the location of each virion or vescicle center has been determined, the segmentation can be calculated.

* Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-secondary` block) and select :badge:`Segmentation (closed surfaces),badge-primary`

* Set the desired `Segmentation radius tolerance`

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Segmentation (closed surfaces),badge-primary` block and select the **Segmentation** tab to inspect the segmentation results

``nextPYP`` calculates implicit representations of surfaces and uses a single threshold to determine the segmentation result

* (optional) In many cases, the default value for the threshold gives reasonable results, but users can also select a different threshold by selecting a virion from the table to show its 3D slices and the segmentation thresholds (8 different thresholds are shown as yellow contours in columns 1-8). The highlighted column number represents the current threshold selection (default is 1, click on a different column to select a better threshold). If none of the columns look reasonable (or if you want to ignore the current virion), select the last column labeled as "-". Repeat this process for all virions in the tilt-series and all tilt-series in the dataset

Open surfaces (membrain-seg)
----------------------------

``nextPYP`` segments open surfaces using `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_:

* Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Segmentation (open surfaces),badge-primary`

* Select the desired parameters for MemBrain-Seg including the location of a pre-trained model

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Segmentation (open surfaces),badge-primary` block to inspect the results of the segmentation

.. note::

    As of now, open surfaces are only used for visual inspection or for processing outside of ``nextPYP``

.. seealso::

    * :doc:`Denoising<denoising>`
    * :doc:`2D particle picking<picking2d>`
    * :doc:`3D particle picking<picking3d>`
    * :doc:`Pattern mining (MiLoPYP)<milopyp>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`