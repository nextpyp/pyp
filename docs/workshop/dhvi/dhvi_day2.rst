#############################
DHVI nextPYP tutorial (day 2)
#############################

Part 1: Geometry-based picking (HIV-1 Gag)
==========================================

.. nextpyp:: Geometry-based picking
  :collapsible: open
    
  * We will be utilizing three separate blocks to perform geometrically constrained particle picking. This will allow for increased accruacy in particle detection and provides geometric priors for downstream refinement. 
  
  * Block 1: Virion selection
  
    * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

    * Go to the **Particle detection** tab:
      
      - Set ``Detection method`` to virions

      - Set ``Virion radius (A)`` to 500 (half the diameter we measured)
      
    * Click :bdg-primary:`Save`

  * Block 2: Virion segmentation

    * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

    * Click :bdg-primary:`Save`

  * Block 3: Spike (Gag) detection
  
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

  * **First block (unfiltered):**

    * Click on ``Tomograms`` (output of the **first** :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Segmentation (membrain/tardis)`

    * On the **Tomogram segmentation** tab:

      - Set ``Method`` to *"membrain-seg (membranes)"*

        - Enable ``Pre-process tomograms``

          - Set ``Pixel size rescaling`` to 11

          - Enable ``Deconvolution filter``

      - Set the ``Pre-trained model (*.ckpt)`` to the following file: ``/nfs/bartesaghilab/membrain-seg-models/MemBrain_seg_v10_alpha.ckpt``.

      - Enable ``Test time augmentation``

      - Set ``Sliding window size`` to 96

  * **Second block (filtered):**

    Use the same parameters as the previous block except:

    * On the **Tomogram reconstruction** tab:

      - Set ``Filter connected components`` to *"by number"*

        - Set ``Components to keep`` to 10

      - Set ``Thickness of slab to keep (unbinned voxels)`` to 1228

  * **Example segmentation:**

    .. md-tab-set::

      .. md-tab-item:: Unfiltered

        .. figure:: ../images/surface_milo/segmentation_unfiltered.webp

      .. md-tab-item:: Filtered

        .. figure:: ../images/surface_milo/segmentation_filtered.webp

.. nextpyp:: Step 2: Improving segmentations
  :collapsible: closed

  Sometimes the segmentations might not be as good as preferred. This mainly happens in two different ways. Either an actual membrane does not get segmented or non-membrane features get segmented. It is possible to experiment with different parameters to improve the results.

  * **Pre-processing block:**

    Sometimes the parameters that make the particles more prominent actually hurt the segmentation. Because of this, as we did in this tutorial, it can be useful to try different reconstruction methods for segmentation. ``imod`` is usually a safe choice, with ``hamming`` high-frequency filtering usually improving things even further. Another choice can be to use ``fakeSIRT`` with higher number of iterations than the block used for particle picking.

  * **Segmentation block:**

    Our suggestions for the segmentation parameters are:

    * **Pre-processing:**

      - Pixel size rescaling:

        It is a good idea to try values between 10 to 12 if your tomograms have different effective pixel sizes, as the membrain-seg models are trained with this range.

      - Deconvolution filter:

        This can help if the SNR of the tomograms is low. Try changing its parameters or disabling if it causes artifacts.

    * **Filter connected components:**

      - By number:

        This is usually a good choice, especially since it can filter out large connected components, which are usually reconstruction artifacts or carbon edges, while also removing smaller components that are not important.

      - By size:

        This can be a good choice if the membranes have a relatively fixed size or there are too many small artifacts.

    * **Thickness of slab to keep (unbinned voxels):**

      This can be very useful at filtering out artifacts since most of the artifacts span close to the lower and upper ends of the tomogram. You should choose the smallest thickness you can that would not remove too many features you actually want to keep. In fact, it can be a good idea to keep the actual tomogram thickness slightly larger than necessary, just to have a better chance for this filtering to work well.

    * **Test time augmentation:**

      This usually helps the segmentations but it makes the processing time longer.

    * **Sliding window size:**

      This can have different effects depending on the dataset. While it is better to try different values, we find 96 to be good for this dataset. Lowering this value also reduces the memory requirements.

Part 3: Moleuclar pattern mining: exploration phase
===================================================

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

      After we run the block, we can see that we initially have 4,937 patches.

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP Model`` (output of the **first** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 30

      .. tip::
        If you want to see metrics to evaluate the performance of the model:

        * On the **Pattern mining** tab:

          - Enable ``Calculate metrics``

        * On the **Resources** tab:

          - Set ``Logging level`` to *"diagnostic (debug)"*

        Then you will be able to see many different metrics in the logs. Refer to the end of this tutorial for their descriptions.

      .. figure:: ../images/surface_milo/iteration_1_embedding.webp

        UMAP embedding

      We can see some patches with the spike proteins in the UMAP embedding but not all of them are close to each other. Meanwhile, bad patches such as the ones centered at the virion, membrane, carbon edge, background, or any other unwanted features seem easier to cluster.

      .. figure:: ../images/surface_milo/iteration_1_labels.webp

        Cluster embeddings

      Here we can see the clusters. Out of all these clusters, we want to keep only the ones that have a membrane at the bottom. However, we need to be conservative at filtering, as we do not want to filter out a cluster that might have good patches this early. We can look at these cluster embeddings to better understand how close the clusters are, which can help us not remove clusters that are too close to a good clusters. However, in general, the cluster matrix is the main tool we will use (the whole matrix is not shown due to its large size).

      .. figure:: ../images/surface_milo/iteration_1_matrix_example.webp

        Example clusters

      Here, we can see that the cluster 15 mostly has patches centered inside the membrane while 16 mostly has background. This means we would prefer to filter them out. Cluster 13 seems to have bad patches as well. However, it also has some patches that seem to be centered at spike proteins. Since we can eliminate bad patches in a future iteration but we cannot recover filtered out good patches, we will act conservatively and keep this cluster.

      To not waste too much time, it can be easier to simply keep all clusters that have a membrane at the bottom, without looking at the center. After checking all rows of the matrix, we decide to only keep these clusters::

        2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 19, 23, 27, 28

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

        After we run the block, we can see that we decreased our patch count from 4,937 to 1,919.

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP Model`` (output of the **second** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 40

      .. figure:: ../images/surface_milo/iteration_2_embedding.webp

        UMAP embedding

      We can see that we eliminated a significant amount of bad patches and the good patches take up much more space. There are still some patches that do not seem to have a membrane at the bottom, but they are very scarce, which makes it difficult to filter them out. It is possible to choose a few clusters if filtering out a few true positives is not considered important. However, we will instead skip filtering for this iteration and zoom in to the patch centers in the next iteration.

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

        After we run the block, we can see that our patch count is still 1,919, as we did not do any filtering.

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP model`` (output of the **third** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern Mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 50

      .. figure:: ../images/surface_milo/iteration_3_embedding.webp

        UMAP embedding

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

        After we run the block, we can see that we further decreased our patch count from 1,919 to 1,159.

    .. md-tab-item:: Evaluation

      * Click on ``MiLoPYP model`` (output of the **fourth** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

      * On the **Pattern mining** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Clusters`` to 60

      .. figure:: ../images/surface_milo/iteration_4_embedding.webp

        UMAP embedding

      We successfully eliminated all patches that were clearly bad. We can still see some patches with no obvious spike protein, but it's not easy to discriminate if they are actually background or just faint or noisy spikes. Nevertheless, we get to filter clusters once more without doing one more iteration before moving onto the refinement phase, so we try to eliminate as much as we can, including patches with carbon edges and artifacts.

      After checking all rows of the matrix, we decide to filter out these clusters::

        6, 7, 15, 17

We did 4 iterations for the exploration phase in this tutorial. However, the strategies can be changed depending on the preferred trade off between time and accuracy. If the refinement phase is going to be used, then it is less problematic to delete some amount of true positives or keep small amount of false positives. If not, then it is likely better to be more careful during the iterations, and possibly change the parameters such as ``min/max distance`` or ``DoG sizes`` to start with more candidate locations.

Part 3: Moleuclar pattern mining: refinement phase
==================================================

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

      After we run the block, we can see that we further decreased our patch count from 1,159 to 1,083.

      This block usually takes a while to run. You can increase or decrease the epoch count or the patch downscaling based on available time.

    .. md-tab-item:: Evaluation

      * Click on ``Particles model`` (output of the :bdg-secondary:`Particle picking (train)` block) and select :bdg-primary:`Particle picking (eval)`

      Use the same parameters as the previous block except:

      * On the **Training/Evaluation** tab:

        - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

        - Set ``Particle radius (A)`` to 80

        - Set ``Threshold for soft/hard positives`` 0.85



Part 3: Denoising with IsoNet2
==============================

This guide explains the workflow for tomogram denoising using IsoNet2 on tilt-series from HA spikes on Influenza viruses. nextPYP support many other algorithm for denoising, such as Topaz-Denoise, IsoNet1, Map2Noise, and CryoCARE. The workflow for these other algorithms is similar to IsoNet2. You can find more information about these algorithms in the nextPYP documentation.


Denoising
----------

.. nextpyp:: Step 1: Generate half-maps
  :collapsible: open

  * Navigate to the ``Edit`` menu from the :bdg-secondary:`Pre-processing` block

  * On the **Tomogram reconstruction** tab:

    - Enable ``Generate half-tomograms``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel


.. nextpyp:: Step 2: Training
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

.. nextpyp:: Step 3: Evaluation
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

    - ``nextPYP`` also supports :doc:`template-search<../guide/picking3d>` and :doc:`molecular pattern mining<../guide/milopyp>`

  * Molecular pattern mining with MiLoPYP

  * Tomogram denosing

      - ``nextPYP`` also supports :doc:`tomogram denoising<../guide/denoising>` using cryoCARE, IsoNet and Topaz Denoise

  :doc:`On day 3<dhvi_day3>` we will demonstrate ``nextPYP``'s functionality for 3D refinement.