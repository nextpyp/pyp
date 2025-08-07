=====================
Tomogram segmentation
=====================

Closed surfaces
---------------

``nextPYP`` can segment *closed* surfaces (such as virions or spherical vescicles) using an energy-based algorithm described in `Bartesaghi et al. (2005) <https://cryoem.cs.duke.edu/node/energy-based-segmentation-of-cryo-em-tomograms/>`_. Segmentation proceeds in two steps:

Find centers
~~~~~~~~~~~~

The first step is to locate a marker inside each virion or vescicle, which can be done using any of the particle picking methods implemented in ``nextPYP``:

* Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle Picking`

* Select the desired particle picking algorithm and corresponding parameters, see :doc:`Particle picking<picking3d>`. The "virion" picking method is especially designed to find the center of spherical virions and estimate their radius (which will be useful later)

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-primary:`Particles` tab to inspect the results of the detection

Segment surfaces
~~~~~~~~~~~~~~~~

Once the location of each virion or vescicle center has been determined, the segmentation can be calculated:

* Click on ``Particles`` (output of the :bdg-secondary:`Particle-Picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

* Set the desired `Segmentation radius tolerance`

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-secondary:`Segmentation (closed surfaces)` block and select the **Segmentation** tab to inspect the results

``nextPYP`` calculates implicit representations of surfaces and uses a single threshold to determine the location of the membranes

* (optional) In cases where the default value for the threshold gives innacurate results, users can select a different threshold by selecting a virion from the table to show its 3D slices and the segmentation thresholds (8 different thresholds are shown as yellow contours in columns 1-8). The highlighted column number represents the current threshold selection (default is 1, click on a different column to select a better threshold). If none of the columns look reasonable (or if you want to ignore the current virion), select the last column labeled as "-". Repeat this process for all virions in the tilt-series and all tilt-series in the dataset

Open surfaces and filaments
---------------------------

``nextPYP`` can also segment *open* surfaces, actin and microtubules using the packages `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_ and `TARDIS <https://github.com/SMLC-NYSBC/TARDIS>`_:

* Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Segmentation (open surfaces, filaments)`

* Select the desired segmentation method and corresponding parameters. For example, if running membrain-seg, specify the location of a pre-trained model (``*.ckpt``) downloadable from their `Github repository <https://github.com/teamtomo/membrain-seg>`_

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-primary:`Segmentation (open surfaces, filaments)` block to inspect the results of the segmentation

.. note::

    The segmented ``*.rec`` volumes are saved in the project directory under the folder ``mrc/`` 