===================
3D particle picking
===================

``nextPYP`` provides multiple methods for particle picking.

Option 1: Import, Manual or Size-based
======================================

#. Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

#. Select one detection method from: "manual", "import", "size-based" and "spherical"

Import coordinates
------------------

``nextPYP`` can import extrernal particles that are saved as IMOD models format

* Select "import" as the ``Detectiom method``

* Set the ``Particle radius (A)`` (only used to size the particle markers)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking,badge-primary` block and confirm that the coordinates were imported correctly

.. tip::

    After the run is finished, the total number of particles in the dataset will be shown in the project information area (top left of the page)

Manual picking
--------------

* Select "manual" as the ``Detectiom method``

* Set the ``Particle radius (A)`` (only used to set the radius of the particle markers)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking,badge-primary` block and select particles by clicking on them in the **Tomogram Slices** panel. Go to other tomograms using the navigation bar, and add particles as needed

.. tip::

    - You can remove particles by right-clicking on them
    - Coordinates are saved automatically every time you add or delete a particle
    - To copy an existing list, you simply copy the entire :badge:`Particle-Picking,badge-primary` block

No further action is need for manual particle picking

Size-based picking
------------------

* Select "size-based" as the ``Detectiom method``

* Set the ``Particle radius (A)`` and other parameters

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking,badge-primary` block and inspect the results

Spherical picking
-----------------

* Select "virions" as the ``Detectiom method``

* Set the ``Virion radius (A)`` and other parameters

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking,badge-primary` block and inspect the results

.. tip::

    To manually edit the results, first create a copy of the :badge:`Particle-Picking,badge-primary` block using the "Copy" function. Select ``Copy files and data`` and ``Make automatically-picked particles editable``, then click :badge:`Next,badge-primary`. Once the copy is done, you can navigate to the new block and add/delete particles as needed

.. note::

    Different from the other approaches, this method also estimates the radius of each virion which will be used for other tasks downstream

Option 2: Geometry-based
========================

This method requires the use of three blocks: :badge:`Particle-Picking,badge-primary`, :badge:`Segmentation (closed surfaces),badge-primary`, and :badge:`Particle-Picking (closed surfaces),badge-primary`

#. Detection of virion centers
------------------------------

Virions centers can be detected using any of the available particle picking methods in ``nextPYP``, for example:

*  Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Particle-Picking,badge-primary`

* Select "virions" as the Detection method and specify the approximate radius in A

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking,badge-primary` block and inspect the results


#. Virion segmentation
----------------------

Virion segmentation consists in finding a closed surface around the virion centers that follows the membrane density

* Click on :guilabel:`Particles` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Segmentation (closed surfaces),badge-primary`

* The only parameter required here is the tolerance radius, which limits the segmentations to be within a certain range from the estimated virion radius

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Segmentation (closed surfaces),badge-primary` block and go to the **Segmentation** tab to inspect the results

``nextPYP`` simplifies this process by calculating an implicit representation of the surface that only requires specifying one of several threshold values to detect the virion membrane. In many cases, the default value for the threshold gives reasonable results, but users can also manually select different thresholds using the web-based GUI

Select a virion from the table to show its 3D slices and the segmentation thresholds (8 different thresholds are shown as yellow contours in columns 1-8). The highlighted column number represents the current threshold selection (default is 1, click on a different column to select a better threshold). If none of the columns look reasonable (or if you want to ignore the current virion), select the last column labeled as "-"

* Repeat this process for all virions in the tilt-series and all tilt-series in the dataset

.. figure:: ../images/tutorial_tomo_pre_process_segmentation.webp
    :alt: Virion segmentation

.. note::

    The virion threshold selections are saved automatically every time you click on a column

#. Particle picking from virion surfaces
========================================

Particles attached to the surface of virions detected using the :badge:`Particle-Picking (closed surfaces),badge-primary` block

* Click on :guilabel:`Segmentation (closed)` (output of the :badge:`Segmentation (closed surfaces),badge-secondary` block) and select :badge:`Particle-Picking (closed surfaces),badge-primary`

* Select the method and parameters

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking (closed surfaces),badge-primary` block and inspect the results


Option 3: Neural-network picking
================================

This method requires the use of the :badge:`Particle-Picking (train),badge-primary` and :badge:`Particle-Picking (eval),badge-primary` blocks

Model training
--------------

* Click on :guilabel:`Particles (closed)` (output of any particle picking block, e.g., :badge:`Particle-Picking (closed surfaces),badge-primary`, :badge:`Particle-Picking (closed surfaces),badge-primary`, or :badge:`MiLoPYP (eval,badge-primary`) and select :badge:`Particle-Picking (train),badge-primary`

* Select the parameters for training

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking (train),badge-primary` block to inspect the results

The results of each training run will be saved in a separate folder named ``train/YYYYMMDD_HHMMSS`` and will consist of files in the `.pth` format

Model evaluation
----------------

* Click on :guilabel:`Particles Model` (output of the :badge:`Particle-Picking (train),badge-primary` block) and select :badge:`Particle-Picking (eval),badge-primary`

* Select the location of the ``Trained model (*.pth)`` using the file browser and adjust evaluation parameters as needed. The file browser's default location is the ``train/`` folder from the parent block

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Once the run finishes, navigate to the :badge:`Particle-Picking (eval),badge-primary` block to inspect the results

.. note::

    This block can also be used to detect particles distributed along fibers or tubules by clicking on the ``Use fiber mode`` option and setting the neccessary parameters

.. seealso::

    * :doc:`Neural-network picking<neural_network>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`