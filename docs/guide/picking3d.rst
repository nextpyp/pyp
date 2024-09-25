===================
3D particle picking
===================

``nextPYP`` provides a suite of methods for particle picking that includes size-based, geometry-based and neural network-based methods. It also features an interactive interface to pick particles manually and import coordinates from external programs.


Import, manual, and size-based picking
======================================

These three methods are implemented in the :badge:`Particle-Picking,badge-primary` block which takes :guilabel:`Tomograms` as input.

1. Import coordinates
---------------------

``nextPYP`` can import particle coordinates saved as IMOD models (``*.spk``) or xyz coordinates in plain text format (``*.box``):

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "import" as the ``Detectiom method``

#. Select the location to import the coordinates from

#. Set the ``Particle radius (A)`` (for visualization purposes)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Picking,badge-primary` block and confirm that the coordinates were imported correctly

.. tip::

    The total number of particles in the dataset will be shown in the project information area at the top of the page

2. Manual picking
-----------------

``nextPYP`` also provides a user-friendly UI to quickly pick particles from datasets with up to thousands of tomograms:

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "manual" as the ``Detection method``

#. Set the ``Particle radius (A)`` (for visualization purposes)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Picking,badge-primary` block, inspect different tomogram slices using the slider and select particles by clicking on them. Go to other tomograms using the navigation bar at the top of the page and add particles as needed

.. tip::

    - You can remove particles by right-clicking on them
    - Coordinates are saved automatically every time you add or delete a particle

3. Size-based picking
---------------------

This method described in `Jin et al., JSB (2024) <https://cryoem.cs.duke.edu/node/accurate-size-based-protein-localization-from-cryo-et-tomograms/>`_ can be used to detect particles based on their size. It works both on purified and in-situ samples:

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "size-based" as the ``Detection method``

#. Set the ``Particle radius (A)`` and other parameters as needed (see the :doc:`classification tutorial<../tutorials/class_empiar_10304>` for an example)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Picking,badge-primary` block to inspect the results

.. tip::

    You can manually edit the results of this or any particle picking method by creating a copy of the :badge:`Particle-Picking,badge-primary` block using the "Copy" function, selecting ``Copy files and data`` and ``Make automatically-picked particles editable``, and then clicking :badge:`Next,badge-primary`. Once the copy is done, you can navigate to the new block and manually add/delete particles

Geometry-based picking
======================

This method described in `Liu et al., Nat Meth (2023) <https://cryoem.cs.duke.edu/node/nextpyp-a-comprehensive-and-scalable-platform-for-characterizing-protein-variability-in-situ-using-single-particle-cryo-electron-tomography/>`_ can detect membrane proteins that are attached to the surface of virions or vesicles. It is composed of three stages:

1. Detection of virion centers
------------------------------

The first step is to estimate the position and the approximate radius of each virion or vesicle:

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "virions" as the ``Detection method``

#. Set the expected ``Virion radius (A)`` and other parameters as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Picking,badge-primary` block to inspect the results

.. tip::

    Virion centers can be obtained using any method for particle picking available in ``nextPYP`` (manual, size-based, neural network-based, etc). Since the virion radius will not be estimated automatically in these cases, the value of ``Virion radius (A)`` will be assigned to each virion

2. Virion segmentation
----------------------

The next step is to segment virions in 3D using methods described in `Bartesaghi et al., IEEE-TIP (2005) <https://cryoem.cs.duke.edu/node/energy-based-segmentation-of-cryo-em-tomograms/>`_:

#. Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-secondary` block) and select :badge:`Segmentation (closed surfaces),badge-primary`

#. Adjust the segmentation parameters as needed (defaults should work fine for 10164, for example)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Segmentation (closed surfaces),badge-primary` block and go to the **Segmentation** tab to inspect the results

``nextPYP`` calculates segmentations using implicit surface representations that require specifying a threshold value to uniquely define the detected surface. To facilitate the threshold selection process for each virion, the user can choose from 8 different threshold values (represented as columns in a table). If none of the columns look reasonable (or if a virion should be ignored), the last column labeled as "-" should be selected. This process can be repeated for all virions in a tilt-series and for all tilt-series in the dataset

.. note::

    The selection of virion thresholds is saved automatically every time a column is clicked

Here is a screenshot of the user interface for virion segmentation:

.. figure:: ../images/tutorial_tomo_pre_process_segmentation.webp
    :alt: Virion segmentation

3. Particle picking from virion surfaces
----------------------------------------

The last step is to pick particles from the surface of virions:

#. Click on :guilabel:`Segmentation (closed)` (output of the :badge:`Segmentation (closed surfaces),badge-secondary` block) and select :badge:`Particle-Picking (closed surfaces),badge-primary`

#. Select the particle detection ``Method`` and corresponding parameters. "uniform" is used to select uniformly spaced positions on the surfaces, while "template search" is used to search for positions on the surface that have high-correlation with an external template (provided as an ``*.mrc`` file)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Picking (closed surfaces),badge-primary` block to inspect the results


Neural-network picking
======================

This method described in `Huang et al., ECCV (2022) <https://cryoem.cs.duke.edu/node/accurate-detection-of-proteins-in-cryo-electron-tomograms-from-sparse-labels/>`_ uses consistency regularization to minimize the number of annotations and speedup training:

1. Model training
-----------------

The first step is to obtain a set of particles using any of the methods implemented in the :badge:`Pre-processing,badge-primary` or the :badge:`MiloPYP (eval),badge-primary` blocks to train the neural network:

#. Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-primary` or :badge:`Particle-Picking (closed surfaces),badge-primary` blocks), or on :guilabel:`MiLoPYP Particles` (output of the :badge:`MiLoPYP (eval,badge-primary`) block) and select :badge:`Particle-Picking (train),badge-primary`

#. Adjust the parameters for training as needed. If using MiLoPYP particles, see instructions on how to set parameters :doc:`here<milopyp>`

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Picking (train),badge-primary` block to inspect the training loss

.. note::
    
    * 40-50 particles from 2-3 tomograms are usually sufficient to successfully train a model
    * Trained models for each run will be saved in the project folder under ``train/YYYYMMDD_HHMMSS/*.pth``

2. Model evaluation
-------------------

Once the model has been trained, it can be evaluated to pick particles on the entire dataset:

#. Click on :guilabel:`Particles Model` (output of the :badge:`Particle-Picking (train),badge-primary` block) and select :badge:`Particle-Picking (eval),badge-primary`

#. Select the location of the ``Trained model (*.pth)`` using the file browser and adjust the evaluation parameters as needed (the file browser's default location will be the ``train/`` folder from the parent block)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run completes, navigate to the :badge:`Particle-Picking (eval),badge-primary` block to inspect the particle picking results

.. tip::

    * To improve accuracy, the model can be re-trainined using more labels
    * To detect particles distributed along fibers or tubules, select ``Fiber mode``. This will group neighboring particles, fit a smooth trajectory to them, and re-sample positions along the fitted curve

.. seealso::

    * :doc:`2D particle picking<picking2d>`
    * :doc:`Pattern mining (MiLoPYP)<milopyp>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`