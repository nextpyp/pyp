=====================
Tomogram segmentation
=====================

Closed surfaces
---------------

``nextPYP`` can segment *closed* surfaces (such as virions or spherical vescicles) using an energy-based algorithm described in `Bartesaghi et al. (2005) <https://cryoem.cs.duke.edu/node/energy-based-segmentation-of-cryo-em-tomograms/>`_. Segmentation proceeds in two steps:

Find centers
~~~~~~~~~~~~

The first step is to locate a marker inside each virion or vescicle, which can be done using any of the particle picking methods implemented in ``nextPYP``:

* Click on ``Tomograms`` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle Picking,badge-primary`

* Select the desired particle picking algorithm and corresponding parameters, see :doc:`Particle picking<picking3d>`. The "virion" picking method is especially designed to find the center of spherical virions and estimate their radius (which will be useful later)

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Particles,badge-primary` tab to inspect the results of the detection

Segment surfaces
~~~~~~~~~~~~~~~~

Once the location of each virion or vescicle center has been determined, the segmentation can be calculated:

* Click on ``Particles`` (output of the :badge:`Particle-Picking,badge-secondary` block) and select :badge:`Segmentation (closed surfaces),badge-primary`

* Set the desired `Segmentation radius tolerance`

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Segmentation (closed surfaces),badge-primary` block and select the **Segmentation** tab to inspect the segmentation results

``nextPYP`` calculates implicit representations of surfaces and uses a single threshold to determine the segmentation result

* (optional) In many cases, the default value for the threshold gives reasonable results, but users can also select a different threshold by selecting a virion from the table to show its 3D slices and the segmentation thresholds (8 different thresholds are shown as yellow contours in columns 1-8). The highlighted column number represents the current threshold selection (default is 1, click on a different column to select a better threshold). If none of the columns look reasonable (or if you want to ignore the current virion), select the last column labeled as "-". Repeat this process for all virions in the tilt-series and all tilt-series in the dataset

Open surfaces (membrain-seg)
----------------------------

``nextPYP`` can also segment *open* surfaces using `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_:

* Click on ``Tomograms`` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Segmentation (open surfaces),badge-primary`

* Select the desired parameters for MemBrain-Seg, including the location of a pre-trained model (``*.ckpt``) downloadable from `github <https://github.com/teamtomo/membrain-seg>`_

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Segmentation (open surfaces),badge-primary` block to inspect the results of the segmentation

.. note::

    The segmented ``*.mrc`` volumes will be saved in the project directory under the folder ``mrc/`` 