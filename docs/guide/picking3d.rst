===================
3D particle picking
===================

``nextPYP`` provides a suite of methods for picking particles in 3D, including size-based, geoemtry-based and neura network-based picking. It also provides an interactive interface to pick particles manually and to import coordinates from other programs.


Import, manual, size-based picking
==================================

These methods are implemented in the :badge:`Particle-Picking,badge-primary` block.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary` to create a new particle picking block

Import coordinates
------------------

``nextPYP`` can import external particles saved as IMOD models (``*.spk``) or xyz coordinates in plain text format (``*.box``):

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "import" as the ``Detectiom method``

#. Set the ``Particle radius (A)`` (for visualization purposes)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block and confirm that the coordinates were imported correctly

.. tip::

    After the run is finished, the total number of particles in the dataset will be shown in the project information area at the top of the page

Manual picking
--------------

``nextPYP`` provides a user-friendly UI to easily pick particles from many tomograms:

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "manual" as the ``Detectiom method``

#. Set the ``Particle radius (A)`` (for visualization purposes)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block and select particles by clicking on them. Go to other tomograms using the navigation bar and add particles as needed

.. tip::

    - You can remove particles by right-clicking on them
    - Coordinates are saved automatically every time you add or delete a particle

Size-based picking
------------------

This method described in `Jin et al., JSB (2024)<https://cryoem.cs.duke.edu/node/accurate-size-based-protein-localization-from-cryo-et-tomograms/>`_ works very effectively on purified samples as well as large complexes imaged in-situ:

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "size-based" as the ``Detectiom method``

#. Set the ``Particle radius (A)`` and other parameters as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block to inspect the results

.. tip::

    To manually edit the results of a particle picking method: create a copy of the :badge:`Particle-Picking,badge-primary` block using the "Copy" function, select ``Copy files and data``, select ``Make automatically-picked particles editable``, and then click :badge:`Next,badge-primary`. Once the copy is done, you can navigate to the new block and manually add/delete particles

Geometry-based picking
======================

This method described in `Liu et al., Nat Meth (2023)<https://cryoem.cs.duke.edu/node/nextpyp-a-comprehensive-and-scalable-platform-for-characterizing-protein-variability-in-situ-using-single-particle-cryo-electron-tomography/>`_ is useful to detect particles that are attached to surfaces such as virions or vesicles. It has three stages: virion detection, virion segmentation and constrained particle picking:

Detection of virion centers
---------------------------

The first step is to estimate the position and the approximate radius of each virion:

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "virions" as the ``Detectiom method``

#. Set the ``Virion radius (A)`` and other parameters as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block to inspect the results

.. tip::

    Virion centers can also be obtained using any other method for particle picking available in ``nextPYP`` (manual, size-based, neural network-based, etc). Since the virion radius will not be estimated automatically in this case, the value of ``Virion radius (A)`` will be used instead

Virion segmentation
-------------------

The next step is to segment each virion in 3D using methods described in `Bartesaghi et al., IEEE-TIP (2005)<https://cryoem.cs.duke.edu/node/energy-based-segmentation-of-cryo-em-tomograms/>`_:

#. Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-secondary` block) and select :badge:`Segmentation (closed surfaces),badge-primary`

#. Adjust the segmentaton paraemters as needed (defaults should work fine for 10164, for example)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Segmentation (closed surfaces),badge-primary` block and go to the **Segmentation** tab to inspect the results

``nextPYP`` calculates segmentations using implicit surface representations that require specifying a threshold value to uniquely define the surface. To faciliate the threshold selection process for each virion, the user can choose from 8 different threshold values (represented as columns in a table). If none of the columns look reasonable (or if a virion should be ignored), the last column labeled as "-" should be selected. This process can be repeated for all virions in a tilt-series and for all tilt-series in the dataset

.. note::

    The selection of virion thresholds is saved automatically every time a column is clicked

This is a screenshot of the user interface for virion segmentation:

.. figure:: ../images/tutorial_tomo_pre_process_segmentation.webp
    :alt: Virion segmentation

Particle picking from virion surfaces
-------------------------------------

The last step is to pick particles from the surface of virions which is done using the :badge:`Particle-Picking (closed surfaces),badge-primary` block:

#. Click on :guilabel:`Segmentation (closed)` (output of the :badge:`Segmentation (closed surfaces),badge-secondary` block) and select :badge:`Particle-Picking (closed surfaces),badge-primary`

#. Select the particle detection ``Method`` and corresponding parameters. "uniform" is used to select uniformly spaced positions from the surfaces, while "template search" is used to search for positions on the surface that have high-correlation with an external template (provided as an ``*.mrc`` file)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking (closed surfaces),badge-primary` block to inspect the results


Neural-network picking
======================

This method described in `Huang et al., ECCV (2022)<https://cryoem.cs.duke.edu/node/accurate-detection-of-proteins-in-cryo-electron-tomograms-from-sparse-labels/>`_ uses consistency regularization to minimize the number of annotations and speedup training.

Model training
--------------

The first step is to obtain a set of particles using any of the methods implemented in the :badge:`Pre-processing,badge-primary` block (import, manual, size-based, or virions) or the :badge:`MiloPYP (eval),badge-primary` block so we can train the neural network:

#. Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-primary` or :badge:`Particle-Picking (closed surfaces),badge-primary` blocks), or on :guilabel:`MiLoPYP Particles` (output of the :badge:`MiLoPYP (eval,badge-primary`) block) and select :badge:`Particle-Picking (train),badge-primary`

#. Adjust the parameters for training as needed. If using MiLoPYP particles, see instructions on how to set paraemters :doc:`here<milopyp>`

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once finished, navigate to the :badge:`Particle-Picking (train),badge-primary` block to inspect the training loss

.. note::
    
    The trained models for each run will be saved in the project folder as ``train/YYYYMMDD_HHMMSS/*.pth``

Model evaluation
----------------

Once the model has been trained, it can be evaluated on the entire dataset:

#. Click on :guilabel:`Particles Model` (output of the :badge:`Particle-Picking (train),badge-primary` block) and select :badge:`Particle-Picking (eval),badge-primary`

#. Select the location of the ``Trained model (*.pth)`` using the file browser and adjust the evaluation parameters as needed (the file browser's default location will be the ``train/`` folder from the parent block)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking (eval),badge-primary` block to inspect the particle picking results

.. note::

    To detect particles distributed along fibers or tubules, select ``Fiber mode``. This will group neighboring particles, fit a smooth trajectory to them, and re-sample positions along the fitted curve

.. seealso::

    * :doc:`2D particle picking<picking2d>`
    * :doc:`Pattern mining (MiLoPYP)<milopyp>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`