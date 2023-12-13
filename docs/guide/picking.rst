================
Particle picking
================

``nextPYP`` implements several strategies for particle picking from 2D micrographs and 3D tomograms


Method 1: Manual picking (2D and 3D)
====================================

.. tabbed:: 2D

    - Inside the :badge:`Pre-processing,badge-secondary` block, go to the **Micrographs** tab

    - Create a new list by clicking :badge:`New,badge-primary`, typing a name for the list and clicking :badge:`Create,badge-primary`

    - Select particles by clicking on their centers

    - Navigate to other micrographs, add additional particles as needed

.. tabbed:: 3D

    - Inside the :badge:`Pre-processing,badge-secondary` block, go to the **Reconstruction** group in the **Tilt Series** tab

    - Create a new list by clicking :badge:`New,badge-primary`, typing a name for the list and clicking :badge:`Create,badge-primary`

    - Scroll through the tomogram and select particles by clicking on their centers in 3D

    - Navigate to other tilt-series, add additional particles as needed

.. tip::

    - You can remove particles from a list by right-clicking on a marker
    - Particle lists are saved automatically every time you add or delete a particle
    - You can copy an existing list using :badge:Copy,badge-primary`, delete an existing list using :badge:`Delete,badge-primary`, or load a list using :badge:`Load,badge-primary`

Method 2: Size-based picking (2D and 3D)
========================================

This method only requires specifying the radius of your particles:

- On the :badge:`Pre-processing,badge-secondary` block, click on the icon :fa:`bars, text-primary` and choose the :fa:`edit, text-primary` `Edit` option

- Go to the **Particle detection** tab and select `auto` as the ``Detection method``

- Set the value of the ``Detection radius`` to the expected particle size

- (optional) Adjust parameters for particle detection

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary`

This will run particle picking on all micrographs or tilt-series in the dataset and the results will be displayed inside the :badge:`Pre-processing,badge-secondary` block

.. tip::

    After the run is finished, the total number of particles in the dataset will be shown in the project information area (top left of the page)

Method 3: Geometry-based picking (3D only)
==========================================

This mode is designed to be used for picking membrane proteins from the surface of virions or vesicles:

Step 1: Virion detection
------------------------

On the :badge:`Pre-processing,badge-secondary` block, click on the icon :fa:`bars, text-primary`, choose the :fa:`edit, text-primary` `Edit` option, and go to the **Virion/spike detection** tab

There are three modes available for virion picking:

.. tabbed:: Manual

    * We will assume that you already have a list of manually picked virion centers (see Method 1 above)

    * Select `manual` as the ``Virion detection method``

    * Select the list of virion centers from the ``Select list of positions`` drop-down menu (top of the form)

    * Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to detect virions on all tilt-series in the dataset

.. tabbed:: Size-based

    * Select `auto` as the ``Virion detection method``

    * Set the desired ``Virion radius``

    * (optional) Adjust virion picking parameters

    * Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to detect virions on all tilt-series in the dataset

.. tabbed:: Neural-network
    
    * We will assume that you already have a list of manually picked virion centers (see Method 1 above)

    * Select `pyp-train` as the ``Virion detection method``

    * Select the list of virion positions from the ``Select list of positions`` drop-down menu (top of the form)

    * (optional) Go to the **Training/evaluation** tab and adjust the training parameters

    * Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` and wait for the training to finish

    * Go to the **Virion/spike detection** tab

    * Select `pyp-eval` as the ``Virion detection method``

    * Go to the **Training/evaluation** tab and specify the ``Trained model`` obtained in the previous step by clicking on the icon :fa:`search, text-primary` and navigating to the ``train/`` folder inside the :badge:`Pre-processing,badge-secondary` block. Each training run will be saved in a separate folder (named with the timestamp ``YYYYMMDD_HHMMSS``), where multiple intermediate models in `.pth` format will be available.

    * Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to detect virions on all tilt-series in the dataset

Step 2 (optional): Virion segmentation
--------------------------------------

Virion segmentation consists in finding a closed surface around the virion centers that follows the membrane density. ``nextPYP`` simplifies this process by calculating an implicit representation of the surface that only requires specifying one of several threshold values to detect the virion membrane. In many cases, the default value for the threshold gives reasonable results, but users can also manually select different thresholds using the web-based GUI

* Inside the :badge:`Pre-processing,badge-secondary` block, go to the **Tilt-series** tab, and select the **Segmentation** group

* Select a virion from the table to show its 3D slices and the segmentation thresholds (8 different thresholds are shown as yellow contours in columns 1-8). The highlighted column number represents the current threshold selection (default is 1, click on a different column to select a better threshold). If none of the columns look reasonable (or if you want to ignore the current virion), select the last column labeled as "-"

* Repeat this process for all virion in the tilt-series and all tilt-series in the dataset

.. figure:: ../images/tutorial_tomo_pre_process_segmentation.webp
    :alt: Virion segmentation

.. note::

    The virion threshold selections are saved automatically every time you click on a column

Step 3: Spike picking
---------------------

There are two methods for picking spikes on the surface of virions:

.. tabbed:: Constrained template search

  * Set ``Spike detection method`` to `template search`

  * Specify a ``Spike search template`` using the file picker (must be .mrc format). The bottom z-slice of the template will be placed exactly at the membrane plane to carry out the search.

  * (optional) Adjust parameters for the template search

.. tabbed:: Uniformly spaced positions

  * Set ``Spike detection method`` to `uniform`

  * (optional) Adjust the parameters for uniform picking

Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to pick spikes on all virions in the dataset

Inspect the results by clicking inside the :badge:`Pre-processing,badge-secondary` block (**Tilt-series** tab, **Reconstruction** group)

.. tip::

    For well behaved datasets, Steps 1-3 can be run without user input in the same pre-processing run

.. seealso::

    * :doc:`Neural-network picking<neural_network>`
    * :doc:`Filters<filters>`
    * :doc:`Overview<overview>`