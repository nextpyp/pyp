===================
3D particle picking
===================

``nextPYP`` provides three types of methods for particle picking that are implemented using specialized blocks (separate from the pre-procesing block). These include the :badge:`Particle-Picking,badge-primary`, :badge:`Segmentation (closed surfaces),badge-primary`, :badge:`Particle-Picking (closed surfaces),badge-primary`, :badge:`Particle-Picking (train),badge-primary` and :badge:`Particle-Picking (eval),badge-primary` blocks

1: Import, manual, size-based, and virion picking
=================================================

These methods are implemented in the :badge:`Particle-Picking,badge-primary` block.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary` to create a new particle picking block

Import coordinates
------------------

``nextPYP`` can import external particles saved as IMOD models (``*.spk``):

#. Select "import" as the ``Detectiom method``

#. Set the ``Particle radius (A)`` (for visualization purposes)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block and confirm that the coordinates were imported correctly

.. tip::

    After the run is finished, the total number of particles in the dataset will be shown in the project information area at the top of the page

Manual picking
--------------

``nextPYP`` provides a user-friendly UI to easily pick particles from many tomograms:

#. Select "manual" as the ``Detectiom method``

#. Set the ``Particle radius (A)`` (for visualization purposes)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block and select particles by clicking on them. Go to other tomograms using the navigation bar and add particles as needed

.. tip::

    - You can remove particles by right-clicking on them
    - Coordinates are saved automatically every time you add or delete a particle

Size-based picking
------------------

This is a simple method that works very effectively on purified samples as well as large complexes imaged in-situ:

#. Select "size-based" as the ``Detectiom method``

#. Set the ``Particle radius (A)`` and other parameters as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block to inspect the results

Virion picking
--------------

This method estimates the position and the approximate radius of virions (useful for doing segmentation later):

#. Select "virions" as the ``Detectiom method``

#. Set the ``Virion radius (A)`` and other parameters as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block to inspect the results

.. tip::

    To manually edit the results of a particle picking method: create a copy of the :badge:`Particle-Picking,badge-primary` block using the "Copy" function, select ``Copy files and data``, select ``Make automatically-picked particles editable``, and then click :badge:`Next,badge-primary`. Once the copy is done, you can navigate to the new block and manually add/delete particles

2: Geometry-based picking
=========================

This method is useful to detect particles that are attached to surfaces such as virions or vesicles. It has three stages: virion detection, virion segmentation and constrained particle picking (each done using a dedicated block):

Detection of virion centers
---------------------------

Virions centers can be detected using any of the methods available in the :badge:`Particle-Picking,badge-secondary` block, but the "virions" method is most commonly used:

#.  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select "virions" as the ``Detection method`` and specify the approximate radius in A

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking,badge-primary` block and inspect the results

Virion segmentation
-------------------

The next step is to segment each virion in 3D:

#. Click on :guilabel:`Particles` (output of the :badge:`Particle-Picking,badge-secondary` block) and select :badge:`Segmentation (closed surfaces),badge-primary`

#. The only parameter required here is the ``Segmentation radius tolerance``, which limits the segmentation to be within a range of the estimated virion radius

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


3: Neural-network picking
=========================

This method has two stages (training and evaluation) and uses the :badge:`Particle-Picking (train),badge-primary` and :badge:`Particle-Picking (eval),badge-primary` blocks

Model training
--------------

The first step is to train the neural network:

#. Click on :guilabel:`Particles` (output of any of the particle picking blocks, e.g., :badge:`Particle-Picking,badge-primary`, :badge:`Particle-Picking (closed surfaces),badge-primary`, or :badge:`MiLoPYP (eval,badge-primary`) and select :badge:`Particle-Picking (train),badge-primary`

#. Select the parameters for training

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once finished, navigate to the :badge:`Particle-Picking (train),badge-primary` block to inspect the training loss

.. note::
    
    The trained models for each run will be saved as ``train/YYYYMMDD_HHMMSS/*.pth``

Model evaluation
----------------

Once the model has been trained, it can be avaluated on the entire dataset:

#. Click on :guilabel:`Particles Model` (output of the :badge:`Particle-Picking (train),badge-primary` block) and select :badge:`Particle-Picking (eval),badge-primary`

#. Select the location of the ``Trained model (*.pth)`` using the file browser and adjust the evaluation parameters as needed (the file browser's default location will be the ``train/`` folder from the parent block)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking (eval),badge-primary` block to inspect the particle picking results

.. note::

    To detect particles distributed along fibers or tubules, use the ``Use fiber mode`` option and set the corresponding parameters as needed

.. seealso::

    * :doc:`2D particle picking<picking2d>`
    * :doc:`Pattern mining (MiLoPYP)<milopyp>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`