#####################
DHVI workshop (day 2)
#####################

Session 1: Geometry-based picking
=================================

In this session, we will show how to automate the picking of **HIV-1 Gag** from the surface of immature virions. We will also extract the normal orientation at the location of each particle to facilitate 3D refinement. 

Before we start, we will measure the diameter of the virions using the measure tool in the :bdg-secondary:`Pre-processing` block.
  
.. nextpyp:: Step 1: Virion selection
  :collapsible: open
  
  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Go to the **Particle detection** tab:
    
    - Set ``Detection method`` to *virions*

    - Set ``Virion radius (A)`` to 500 (value obtained using measure tool)
    
  * Click :bdg-primary:`Save`

.. nextpyp:: Step 2: Virion segmentation
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

  * We will leave all default settings here

  * Click :bdg-primary:`Save`

.. nextpyp:: Step 3: Gag protein detection
  :collapsible: open

  * Click on ``Segmentation (closed)`` (output of the :bdg-secondary:`Segmentation (closed surfaces)` block) and select :bdg-primary:`Particle-Picking (closed surfaces)`
  
  * Go to the **Particle detection** tab:
    
    - Set ``Particle radius (A)`` to 50

    - Set ``Detection method`` to *uniform*

    - Set ``Size of equatorial band to restrict spike picking (A)`` to 800
    
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel

This will produce a total of approximately 12k particles.

``nextPYP`` supports many other methods for :doc:`3D particle picking<../guide/picking3d>`, including size-based, 3D template-matching, etc.

Session 2: Molecular pattern mining
===================================

In this session, we will demonstrate how to use MiLoPYP-2, a new membrane-aware framework designed to facilitate particle picking from irregular and more challenging samples. The basic workflow is composed of two stages: 

1) an **exploration phase** to identify particles of interest
2) a **refinement phase** to accurately pick all instances of the identified target

Part 1: MiLoPYP-2 exploration
-----------------------------

In the exploration phase, we use the segmentations we calculated yesterday to automatically filter out the candidate particle locations that are too close or too distant to any surface. We also orient the patches so that they are normal to the surface, making all the spike proteins aligned.

We also use a feature called **iterative exploration**, which is one of the main strengths of this pipeline. Even after the initial filtering, it can still initially be difficult to cluster all the good patches we want since their numbers are usually quite low compared to bad patches. Luckily, it is usually much easier to cluster bad patches. We use this to our advantage by doing multiple iterations of exploration, where each iteration removes clusters with bad patches rather than trying to find clusters with only good patches.

.. nextpyp:: Iteration 1
  :collapsible: open

  .. md-tab-set::

    .. md-tab-item:: Training

      * Click on ``Tomograms`` (output of the :bdg-secondary:`Denoising (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

      * On the **Pattern Mining** tab:

        - Set ``Input type`` to *3D only*

        - Set ``Epochs`` to 50

        - Set ``Bounding box (binned voxels)`` to 72

        - Set ``Interval to perform validation`` to 5

        - Enable ``Surface constrained``

          - Set ``Segmentation directory`` to the location of the ``/mrc`` folder in the segmentation block. You can get the path to the segmentation block by clicking ``Show filesystem location`` from the block menu

          - Set ``Use DoG`` to *DoG*

            - Set ``Min distance`` to 5

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP model`` (output of the **first** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate, for example, ``YYYYMMDD_HHMMSS/model_50.pth``where ``YYYYMMDD_HHMMSS`` is the date and time of training.

        - Set ``Clusters`` to 30

.. nextpyp:: Iteration 2
  :collapsible: open

  .. md-tab-set::

    .. md-tab-item:: Training

      * Click on ``MiLoPYP particles`` (output of the **first** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

      * On the **Pattern mining** tab:

        - Set ``Epochs`` to 100

        - Set ``Interval to perform validation`` to 10

        - Enable ``Surface constrained``

          - Set ``Segmentation directory`` to the location of the ``/mrc`` folder in the segmentation block. You can get the path to the segmentation block by clicking ``Show filesystem location`` from the block menu

          - Set ``Use DoG`` to *DoG*

            - Set ``Min distance`` to 5

        - Enable ``Iterate``

        - Set ``Patch coordinate location`` to the location of the ``/train/interactive_info_parquet.gzip`` file in the **first** **evaluation** block.

        - Set ``Class labels`` to a comma separated list of classes that contain spike protein

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP Model`` (output of the **second** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 40

We did 2 iterations for the exploration phase in this tutorial. However, the strategy can be changed depending on the preferred trade off between time and accuracy. In some cases, it may ne necesary to change the parameters ``min/max distance`` or ``DoG sizes`` to start with more candidate locations.

Part 2: MiLoPYP-2 refinement
----------------------------

Refinement phase also uses the segmentations to constrain particle picking to surfaces. This is done by creating a binary mask based on the distance from the segmentation. This binary mask is then used to filter input and output coordinates, in the loss function, or to modify the input tomograms to directly remove any signal outside the mask.

.. nextpyp:: Refinement
  :collapsible: open

  .. md-tab-set::

    .. md-tab-item:: Training

      * Click on ``MiLoPYP particles`` (output of the **last** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`Particle picking (train)`

      * On the **Training/Evaluation** tab:

        - Set ``Coordinates for training`` to *class labels from MiLoPYP*

          - Set ``Class IDs`` to a comma separated list of classes that contain spike protein

        - Set ``Epochs`` to 25

        - Set ``Max number of particles`` to 300

        - Set ``Learning rate`` to 0.001

        - Set ``Validation interval (epochs)`` to 3

        - Enable ``Use masking``

          - Set ``Segmentation directory`` to the location of the ``/mrc`` folder in the segmentation block

        - Enable ``Enable compilation``

          - Set ``Compile mode`` to *max autotune*

    .. md-tab-item:: Evaluation

      * Click on ``Particles model`` (output of the :bdg-secondary:`Particle picking (train)` block) and select :bdg-primary:`Particle picking (eval)`
      * On the **Training/Evaluation** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate, for example, ``YYYYMMDD_HHMMSS/model_25.pth``where ``YYYYMMDD_HHMMSS`` is the date and time of training.

        - Set ``Particle radius (A)`` to 80

Day 2 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  * How to apply geometry-based picking to detect Gag protein from HIV-1 virions

  * Other particle picking methods available in ``nextPYP``
  
  * How to use MiLoPYP-2 to pick spike proteins from SARS-CoV-2 virions

  :doc:`On day 3<dhvi_day3>` we will demonstrate ``nextPYP``'s functionality for 3D refinement.