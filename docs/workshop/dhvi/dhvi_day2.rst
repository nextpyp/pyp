###################
DHVI course (day 2)
###################

Part 1: Geometry-based picking (HIV-1 Gag)
==========================================

We will be utilizing three separate blocks to perform geometrically constrained particle picking. This will allow for increased accruacy in particle detection and provides geometric priors for downstream refinement. 
  
.. nextpyp:: Step 1: Virion selection
  :collapsible: open
  
  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Go to the **Particle detection** tab:
    
    - Set ``Detection method`` to virions

    - Set ``Virion radius (A)`` to 500 (half the diameter we measured)
    
  * Click :bdg-primary:`Save`

.. nextpyp:: Step 2: Virion segmentation
  :collapsible: open

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

  * Click :bdg-primary:`Save`

.. nextpyp:: Step 2: Gag protein detection
  :collapsible: open

  * Click on ``Segmentation (closed)`` (output of the :bdg-secondary:`Segmentation (closed surfaces)` block) and select :bdg-primary:`Particle-Picking (closed surfaces)`
  
  * Go to the **Particle detection** tab:
    
    - Set ``Detection method`` to uniform

    - Set ``Particle radius (A)`` to 50

    - Set ``Size of equatorial band to restrict spike picking (A)`` to 800
    
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel


Part 2: Tomogram segmentation and filtering
===========================================

Segmentation quality is very important and can affect all downstream tasks as we use binary segmentations for surface constraints.

.. nextpyp:: Step 1: Calculating segmentations
  :collapsible: open

  We will again create two different blocks. One of them will not have any filtering while the other one will filter out unwanted connected components. We are doing this because it can be beneficial to keep unfiltered segmentations as filtering can also remove good components.

  * Click on ``Tomograms`` (output of the **first** :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Segmentation (membrain/tardis)`

  * On the **Tomogram segmentation** tab:

    - Set ``Method`` to *"membrain-seg (membranes)"*

      - Enable ``Pre-process tomograms``

        - Set ``Pixel size rescaling`` to 11

        - Enable ``Deconvolution filter``

    - Set the ``Pre-trained model (*.ckpt)`` to the following file: ``/nfs/bartesaghilab/membrain-seg-models/MemBrain_seg_v10_alpha.ckpt``.

    - Enable ``Test time augmentation``

    - Set ``Sliding window size`` to 96

    - Set ``Filter connected components`` to *"by number"*

      - Set ``Components to keep`` to 10

    - Set ``Thickness of slab to keep (unbinned voxels)`` to 1228

Part 3: MiLoPYP exploration
===========================

In the exploration phase, we use the segmentations to automatically filter out the candidate particle locations that are too close or too distant to any surface. We also orient the patches so that they are normal to the surface, making all the spike proteins aligned.

We also use a feature called **iterative exploration**, which is one of the main strengths of this pipeline. Even after the initial filtering, it can still initially be difficult to cluster all the good patches we want since their numbers are usually quite low compared to bad patches. Luckily, it is usually much easier to cluster bad patches (such as background and carbon edges). We use this to our advantage by doing multiple iterations of exploration, where each iteration removes clusters with bad patches rather than trying to find clusters with only good patches.

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
  :collapsible: closed

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

.. nextpyp:: Iteration 3
  :collapsible: closed

  .. md-tab-set::

    .. md-tab-item:: Training

      In this iteration we decrease the patch size to focus more on the centers, since we already eliminated most of the non-membrane patches.

      * Click on ``MiLoPYP particles`` (output of the **second** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

      Use the same parameters as the previous block except:

      * On the **Pattern mining** tab:

        - Set ``Epochs`` to 200

        - Set ``Bounding box (binned voxels)`` to 36

        - Set ``Learning rate`` to 0.001

        - Set ``Interval to perform validation`` to 20

        - Disable ``Remove segmentation``

        - Set ``Patch coordinate location`` to the location of the ``/train/interactive_info_parquet.gzip`` file in the **second** **evaluation** block.

        - Keep ``Class labels`` empty

        - Enable `Exclude labels`

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP model`` (output of the **third** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern Mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 50

      Our task now is trying to eliminate patches by focusing on their centers. Specifically, we want to eliminate patches that have a carbon edge, gold particle, artifact, or simply nothing at all, latter of which can be more difficult.

      When we look at the cluster matrix, we see that there are a lot of clusters that seem to have bad patches but also some patches with spike proteins, which we would like to keep. As such, we have three options:

      #. Stop the iterations and continue to refinement phase with what we have.

      #. Continue iterations while still filtering conservatively, at the cost of having to spend more time.

      #. Continue iterations but become less conservative at filtering, hoping that we do not filter out too many good patches.

      We will do the third option to save time while also showing one more iteration.

      After checking all rows of the matrix, we decide to filter out these clusters::

        0, 1, 3, 5, 9, 12, 16, 17, 20, 21, 24, 25, 28, 32, 33, 35, 38, 39, 41, 49

.. nextpyp:: Iteration 4
  :collapsible: closed

  .. md-tab-set::

    .. md-tab-item:: Training

      In this iteration we decrease the patch size even further to focus on the spike proteins.

      * Click on ``MiLoPYP particles`` (output of the **third** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

      Use the same parameters as the previous block except:

      * On the **Pattern mining** tab:

        - Set ``Bounding box (binned voxels)`` to 24

        - Set ``Patch coordinate location`` to the location of the ``/train/interactive_info_parquet.gzip`` file in the **third** **evaluation** block.

        - Set ``Class labels`` to ``0, 1, 3, 5, 9, 12, 16, 17, 20, 21, 24, 25, 28, 32, 33, 35, 38, 39, 41, 49``

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP model`` (output of the **fourth** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 60

      We successfully eliminated all patches that were clearly bad. We can still see some patches with no obvious spike protein, but it's not easy to discriminate if they are actually background or just faint or noisy spikes. Nevertheless, we get to filter clusters once more without doing one more iteration before moving onto the refinement phase, so we try to eliminate as much as we can, including patches with carbon edges and artifacts.

      After checking all rows of the matrix, we decide to filter out these clusters::

        6, 7, 15, 17

We did 4 iterations for the exploration phase in this tutorial. However, the strategies can be changed depending on the preferred trade off between time and accuracy. If the refinement phase is going to be used, then it is less problematic to delete some amount of true positives or keep small amount of false positives. If not, then it is likely better to be more careful during the iterations, and possibly change the parameters such as ``min/max distance`` or ``DoG sizes`` to start with more candidate locations.

Part 3: MiLoPYP refinement
==========================

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


Part 3: Denoising with IsoNet2
==============================

This guide explains the workflow for tomogram denoising using IsoNet2 on tilt-series from HA spikes on Influenza viruses. nextPYP support many other algorithm for denoising, such as Topaz-Denoise, IsoNet1, Map2Noise, and CryoCARE. The workflow for these other algorithms is similar to IsoNet2. You can find more information about these algorithms in :doc:`User guide<../../guide/denoising>`.

.. nextpyp:: Step 1: Generate half-tomograms
  :collapsible: open

  * Navigate to the ``Edit`` menu from the :bdg-secondary:`Pre-processing` block

  * On the **Tomogram reconstruction** tab:

    - Enable ``Generate half-tomograms``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel


.. nextpyp:: Step 2: Train the denoise model
  :collapsible: open

  Since IsoNet2 is a deep learning-based denoising method, it requires training before we can apply to our tomograms.

  * Click on ``Tomograms`` (output of the **first** :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Denoising (train)`

  * On the **Tomogram denoising** tab:

    - Set ``Method`` to *"isonet2"*

      - Enable ``Use masking``

        - Set ``Loss function`` to *Huber*

        - Set ``Learning rate`` to 0.0001

        - Set ``Minimum learning rate`` to 0.0001

        - Set ``B-factor`` to 200

        - Set ``Epochs for training`` to 30

        - Set ``Loss function`` to *Huber*

        - Set ``Learning rate`` to 0.0001

        - Set ``Minimum learning rate`` to 0.0001

        - Set ``Missing wedge weight in loss`` to 100

        - Set ``CTF mode`` to *network*

        - Set ``B-factor`` to 200

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

.. nextpyp:: Step 3: Evaluate the trained model
  :collapsible: open

  The last step is to evaluate the trained denoising model on all the tomograms in the dataset.

  * Click on ``Denoising model`` (output of the :bdg-secondary:`Denoising (train)` block) and select :bdg-primary:`Denoising (eval)`

  * On the **Tomogram denoising** tab:

    - Set ``Method`` to *"isonet2"*

    - Set ``Trained model`` to the output of the :bdg-secondary:`Denoising (train)` block

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

Day 2 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  In this session, we learned about some of the methods available in ``nextPYP`` to pick particles:
  
  * Particle picking (geometrically constrained, size-based, nn-based, manual)

    - ``nextPYP`` also supports :doc:`template-search<../../guide/picking3d>` and :doc:`molecular pattern mining<../../guide/milopyp>`

  * Molecular pattern mining with MiLoPYP

  * Tomogram denosing

    - ``nextPYP`` also supports :doc:`tomogram denoising<../../guide/denoising>` using cryoCARE, IsoNet and Topaz Denoise

  :doc:`On day 3<dhvi_day3>` we will demonstrate ``nextPYP``'s functionality for 3D refinement.