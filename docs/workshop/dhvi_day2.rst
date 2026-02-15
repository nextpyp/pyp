###################
DHVI course (day 2)
###################

Session 1: Geometry-based picking (HIV-1 Gag)
=============================================

We will be utilizing three separate blocks to perform geometrically constrained particle picking. This will allow us to automate the picking of Gag proteins from the surface of virions and extract normal orientations to facilitate 3D refinement. 
  
.. nextpyp:: Step 1: Virion selection
  :collapsible: open
  
  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Go to the **Particle detection** tab:
    
    - Set ``Detection method`` to virions

    - Set ``Virion radius (A)`` to 500 (value obtained using measure tool)
    
  * Click :bdg-primary:`Save`

.. nextpyp:: Step 2: Virion segmentation
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

  * Click :bdg-primary:`Save`

.. nextpyp:: Step 3: Gag protein detection
  :collapsible: open

  * Click on ``Segmentation (closed)`` (output of the :bdg-secondary:`Segmentation (closed surfaces)` block) and select :bdg-primary:`Particle-Picking (closed surfaces)`
  
  * Go to the **Particle detection** tab:
    
    - Set ``Detection method`` to uniform

    - Set ``Particle radius (A)`` to 50

    - Set ``Size of equatorial band to restrict spike picking (A)`` to 800
    
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel

This will produce aproximately 20k particles that can be used for 3D refinement.

Session 2: Molecular pattern mining
===================================

Part 1: MiLoPYP exploration
---------------------------

In the exploration phase, we use the segmentations to automatically filter out the candidate particle locations that are too close or too distant to any surface. We also orient the patches so that they are normal to the surface, making all the spike proteins aligned.

We also use a feature called **iterative exploration**, which is one of the main strengths of this pipeline. Even after the initial filtering, it can still initially be difficult to cluster all the good patches we want since their numbers are usually quite low compared to bad patches. Luckily, it is usually much easier to cluster bad patches. We use this to our advantage by doing multiple iterations of exploration, where each iteration removes clusters with bad patches rather than trying to find clusters with only good patches.

.. nextpyp:: Iteration 1
  :collapsible: open

  .. md-tab-set::

    .. md-tab-item:: Training

      * Click on ``Tomograms`` (output of the **second** :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`MiLoPYP (train)`

      * On the **Pattern Mining** tab:

        - Set ``Input type`` to *"3D only"*

        - Set ``Epochs`` to 50

        - Set ``Bounding box (binned voxels)`` to 72

        - Set ``Learning rate`` to 0.00001

        - Set ``Interval to perform validation`` to 5

        - Enable ``Surface constrained``

          - Set ``Segmentation directory`` to the location of the ``/mrc`` folder in the **filtered** segmentation block. You can get the path to the segmentation block by clicking ``Show filesystem location`` from the block menu

          - Set ``Use DoG`` to *"DoG"*

          - Set ``Min distance`` to 5

          - Set ``Max distance`` to 25

          - Enable ``Remove segmentation``

        - Enable ``Enable Compilation``

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP Model`` (output of the **first** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 30

.. nextpyp:: Iteration 2
  :collapsible: open

  .. md-tab-set::

    .. md-tab-item:: Training

      * Click on ``MiLoPYP particles`` (output of the **first** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

      Use the same parameters as the previous block except:

      * On the **Pattern mining** tab:

        - Set ``Epochs`` to 100

        - Set ``Interval to perform validation`` to 10

        - Enable ``Iterate``

        - Set ``Patch coordinate location`` to the location of the ``/train/interactive_info_parquet.gzip`` file in the **first** **evaluation** block.

        - Set ``Class labels`` to ``2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 19, 23, 27, 28``

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP Model`` (output of the **second** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 40

We did 2 iterations for the exploration phase in this tutorial. However, the strategy can be changed depending on the preferred trade off between time and accuracy. If the refinement phase is going to be used, then it is less problematic to delete some amount of true positives or keep small amount of false positives. If not, then it is likely better to be more careful during the iterations, and possibly change the parameters such as ``min/max distance`` or ``DoG sizes`` to start with more candidate locations.

Part 2: MiLoPYP refinement
--------------------------

Refinement phase also uses the segmentations to constrain particle picking to surfaces. This is done by creating a binary mask based on the distance from the segmentation. This binary mask is then used to filter input and output coordinates, in the loss function, or to modify the input tomograms to directly remove any signal outside the mask.

.. nextpyp:: Refinement
  :collapsible: open

  .. md-tab-set::

    .. md-tab-item:: Training

      * Click on ``MiLoPYP particles`` (output of the **fourth** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`Particle picking (train)`

      * On the **Training/Evaluation** tab:

        - Set ``Coordinates for training`` to *"class labels from MiLoPYP"*

          - Set ``Class IDs`` to ``0,1,2,3,4,5,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59``

        - Set ``Epochs`` to 25

        - Set ``Max number of particles`` to 300

        - Set ``Learning rate`` to 0.001

        - Set ``Validation interval (epochs)`` to 3

        - Enable ``Use masking``

          - Set ``Segmentation directory`` to the location of the ``/mrc`` folder in the **filtered** segmentation block (or **unfiltered** if your filters eliminate too many good components, you do not have many false positives from exploration phase, or you simply do not want to filter too much)

          - Set ``Mask radius`` to 25

        - Enable ``Enable compilation``

          - Set ``Compile mode`` to *"max autotune"*

    .. md-tab-item:: Evaluation

      * Click on ``Particles model`` (output of the :bdg-secondary:`Particle picking (train)` block) and select :bdg-primary:`Particle picking (eval)`
      Use the same parameters as the previous block except:

      * On the **Training/Evaluation** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Particle radius (A)`` to 80

        - Set ``Threshold for soft/hard positives`` 0.85

Day 2 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  * Particle picking methods available in nextPYP
  
  * Geometry-based picking of Gag protein from HIV-1 virions

    - ``nextPYP`` also supports :doc:`template-search<../../guide/picking3d>` and :doc:`molecular pattern mining<../../guide/milopyp>`

  * Molecular pattern mining using MiLoPYP to pick proteins from membranes

  :doc:`On day 3<dhvi_day3>` we will demonstrate ``nextPYP``'s functionality for 3D refinement.