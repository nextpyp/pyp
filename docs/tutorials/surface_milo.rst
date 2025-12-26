======================
Surface Constrained MiLoPYP Tutorial
======================

This tutorial explains the workflow for Surface-Constrained MiLoPYP and demonstrates the full pipeline by detecting spike proteins of SARS-CoV-2.

Dataset
==========
* `EMPIAR-10453 <https://www.ebi.ac.uk/empiar/EMPIAR-10453/>`_

Data Pre-processing
==========

.. nextpyp:: Step 1. Import Raw Tilt-Series
  :collapsible: closed

  * Click :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`

  * On the **Raw data** tab:

    - Set the ``Location`` by clicking on the :fa:`search` icon and browsing to ``/nfs/bartesaghilab/micromon/research-bartesaghilab-05/shared/users/yz533@duke.edu/projects/EMPIAR11462-6kqOmPBTLN9teaY/tomo-preprocessing-eggvDsBJq8Ob2FY9/raw/``

    - Type ``*.mrc`` into the filter box (lower right) and click the :fa:`filter` icon

  * On the **Microscope parameters** tab:

    - Set ``Pixel size (A)`` to 1.329

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 84.8

  * Click :bdg-primary:`Save` and :bdg-primary:`Run` the block.

.. nextpyp:: Step 2. Pre-processing
  :collapsible: closed

  We will create two pre-processing blocks with slightly different parameters as the best parameters for segmentation quality and particle picking can be different.

  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

  * **First block (for segmentation):**

    * On the **Frame alignment** tab::

      - Enable ``No movie frames``

    * On the  **CTF determination** tab:

      - Set ``Max resolution (A)`` to 10

    * On the **Tilt-series alignment** tab:

      - Disable ``Reshape tilt-images into squares``

    * On the **Tomogram reconstruction** tab:

      - Set ``Tomogram thickness (unbinned voxels)`` to 1536

      - Set ``2D filtering`` to *"none"*

      - Enable ``Erase fiducials``

      - Set ``High-frequency filtering`` to *"hamming (as in tomo3d)"*

  * **Second block (for particle picking):**

    Use the same parameters as the previous block except:

    * On the **Tomogram reconstruction** tab:

      - Set ``Radial filtering`` to *"fakeSIRT (mimic SIRT reconstruction)"*

.. nextpyp:: Step 3. Filtering
  :collapsible: closed

  We will only use a subset of the tilt-series in this tutorial via the filtering feature of the :bdg-secondary:`Pre-processing` block as explained in the :doc:`filtering<../guide/filters>` guide.

  The list of tilt series used in this tutorial::
      - TS_049
      - TS_050
      - TS_071
      - TS_121
      - TS_162
      - TS_244
      - TS_271
      - TS_288
      - TS_291
      - TS_297

  This list is chosen randomly, but you are free to hand-pick good tomograms or use less/more.

Segmentation
==========

Segmentation quality is very important and can affect all downstream tasks as we use binary segmentations for surface constraints.

.. nextpyp:: Calculating Segmentations
  :collapsible: closed

  We will again create two different blocks. One of them will not have any filtering while the other one will filter out unwanted connected components. We are doing this because it can be beneficial to keep unfiltered segmentations as filtering can also remove good components.

  * **First block (unfiltered):**

    * Click on ``Tomograms`` (output of the **first** :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Segmentation (membrain/tardis)`

    * On the **Tomogram segmentation** tab:

      - Set ``Method`` to *"membrain-seg (membranes)"*

        - Enable ``Pre-process tomograms``

          - Set ``Pixel size rescaling`` to 11

          - Enable ``Deconvolution filter``

      - Set the ``Pre-trained model (*.ckpt)`` to ``/nfs/bartesaghilab/membrain-seg-models/MemBrain_seg_v10_alpha.ckpt``

      - Enable ``Test time augmentation``

      - Set ``Sliding window size`` to 96

  * **Second block (filtered):**

    Use the same parameters as the previous block except:

    * On the **Tomogram reconstruction** tab:

      - Set ``Filter connected components`` to *"by number"*

        - Set ``Components to keep`` to 10

      - Set ``Thickness of slab to keep (unbinned voxels)`` to 1228

.. nextpyp:: Improving Segmentations
  :collapsible: closed

  Sometimes the segmentations might not be as good as preferred. This mainly happens in two different ways. Either an actual membrane does not get segmented or non-membrane features get segmented. It is possible to experiment with different parameters to improve the results.

  * **Pre-Processing Block:**

    Sometimes the parameters that make the particles more prominent actually hurt the segmentation. Because of this, as we did in this tutorial, it can be useful to try different reconstruction methods for segmentation. ``imod`` is usually a safe choice, with ``hamming`` high-frequency filtering usually improving things even further. Another choice can be to use ``fakeSIRT`` with higher number of iterations than the block used for particle picking.

  * **Segmentation Block:**

    Our suggestions for the segmentation parameters are:

    * **Pre-Processing:**

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

      This can have different effects depending on the dataset. While it is better to try different values, we find 96 to be good for this dataset. Lowering this value also reduces the memory need.

Exploration Phase
==========

In the exploration phase, we use the segmentations to automatically filter out the candidate particle locations that are too close or too distant to any surface. We also orient the patches so that they are normal to the surface, making all the spike proteins aligned.

We also use a feature called **iterative exploration**, which is one of the main strengths of this pipeline. Even after the initial filtering, it can still initially be difficult to cluster all the good patches we want since their numbers are usually quite low compared to bad patches. Luckily, it is usually much easier to cluster bad patches (such as background and carbon edges). We use this to our advantage by doing multiple iterations of exploration, where each iteration removes clusters with bad patches rather than trying to find clusters with only good patches.

.. nextpyp:: Iteration 1 (Training)
  :collapsible: closed

  * Click on ``Tomograms`` (output of the **second** :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`MiLoPYP (train)`

  * On the **Pattern Mining** tab:

    - Set ``Input type`` to *"3D only"*

    - Set ``Epochs`` to 50

    - Set ``Bounding box (binned voxels)`` to 72

    - Set ``Learning rate`` to 0.0001

    - Set ``Interval to perform validation`` to 5

    - Enable ``Surface constrained``

      - Set ``Segmentation directory`` to the location of the ``/mrc`` folder in the **filtered** segmentation block. You can get the path to the segmentation block by clicking ``Show filesystem location`` from the block menu

      - Set ``Use DoG`` to *"DoG"*

      - Set ``Min distance`` to 5

      - Set ``Max distance`` to 25

      - Enable ``Remove segmentation``

    - Enable ``Enable Compilation``

    - Set ``Compile mode`` to *"reduce overhead"*

  After we run the block, we can see that we initially have 5214 patches.

.. nextpyp:: Iteration 1 (Evaluation)
  :collapsible: closed

  * Click on ``MiLoPYP Model`` (output of the **first** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

  * On the **Pattern Mining** tab:

    - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

    - Set ``Clusters`` to 30

  .. tip::
    If you want to see metrics to evaluate the performance of the model:

    * On the **Pattern Mining** tab:

      - Enable ``Calculate metrics``

    * On the **Resources** tab:

      - Set ``Logging level`` to *"diagnostic (debug)"*

    Then you will be able to see many different metrics in the logs. Refer to the end of the tutorial for their descriptions.

  .. figure:: ../images/surface_milo/iteration_1_embedding.webp

    UMAP embedding

  We can see some patches with the spike proteins in the UMAP embedding but not all of them are close to each other. Meanwhile, bad patches such as the ones centered at the virion, membrane, carbon edge, background, or any other unwanted features seem easier to cluster.

  .. figure:: ../images/surface_milo/iteration_1_labels.webp

    Cluster Embeddings

  Here we can see the clusters. Out of all these clusters, we want to keep only the ones that have a membrane at the bottom. However, we need to be conservative at filtering, as we do not want to filter out a cluster that might have good patches. We can look at these cluster embeddings to better understand how close the clusters are, which can help us not remove clusters that are too close to a good clusters. However, in general, the cluster matrix is the main tool we will use (the whole matrix is not shown due to its large size).

  .. figure:: ../images/surface_milo/iteration_1_matrix_example.webp

    Example Clusters

  Here, we can see that the cluster 10 mostly has background patches while 13 mostly has carbon edges. This means we would prefer to filter them out. Cluster 14 seems to mostly have bad patches as well. However, it also has some patches that seem to be centered at spike proteins, such as 4th and 8th patches. We will act conservative and keep this cluster, since even though we could eliminate those bad patches in a future iteration, we cannot recover back good patches.

  To not waste too much time, it is easier to simply keep all clusters that have a membrane at the bottom, without looking for a spike protein. After checking all rows of the matrix, we decide to only keep these clusters::

    1, 2, 5, 6, 8, 9, 11, 12, 14, 17, 18, 19, 21, 23, 25, 26

.. nextpyp:: Iteration 2 (Training)
  :collapsible: closed

  * Click on ``MiLoPYP particles`` (output of the **first** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

  Use the same parameters as the previous block except:

  * On the **Pattern Mining** tab:

    - Set ``Epochs`` to 100

    - Set ``Interval to perform validation`` to 10

    - Enable ``Iterate``

    - Set ``Patch coordinate location`` to the location of the ``/train/interactive_info_parquet.gzip`` file in the **first** **evaluation** block.

    - Set ``Class labels`` to ``1, 2, 5, 6, 8, 9, 11, 12, 14, 17, 18, 19, 21, 23, 25, 26``

    After we run the block, we can see that we decreased our patch count from 5214 to 2401.

.. nextpyp:: Iteration 2 (Evaluation)
  :collapsible: closed

  * Click on ``MiLoPYP Model`` (output of the **second** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

  * On the **Pattern Mining** tab:

    - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

    - Set ``Clusters`` to 40

  .. figure:: ../images/surface_milo/iteration_2_embedding.webp

    UMAP embedding

  We can see that we eliminated a good amount of bad patches and the good patches take up much more space. However, the bad patches still exist, and we can filter them out further. Since good patches seem to be the majority now, this time we look for clusters to eliminate rather than keep like we did last time.

  After checking all rows of the matrix, we decide to filter out these clusters::

    1, 3, 11, 15, 20, 22, 25, 26, 36, 37, 39

.. nextpyp:: Iteration 3 (Training)
  :collapsible: closed

  In this iteration we decrease the patch size to focus more on the centers, since we already eliminated most of the non-membrane patches.

  * Click on ``MiLoPYP particles`` (output of the **second** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

  Use the same parameters as the previous block except:

  * On the **Pattern Mining** tab:

    - Set ``Epochs`` to 150

    - Set ``Bounding box (binned voxels)`` to 54

    - Set ``Learning rate`` to 0.001

    - Set ``Interval to perform validation`` to 15

    - Set ``Patch coordinate location`` to the location of the ``/train/interactive_info_parquet.gzip`` file in the **second** **evaluation** block.

    - Set ``Class labels`` to ``1, 3, 11, 15, 20, 22, 25, 26, 36, 37, 39``

    - Enable `Exclude labels`

    After we run the block, we can see that we further decreased our patch count from 2401 to 1639.

.. nextpyp:: Iteration 3 (Evaluation)
  :collapsible: closed

  * Click on ``MiLoPYP Model`` (output of the **third** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

  * On the **Pattern Mining** tab:

    - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

    - Set ``Clusters`` to 50

  .. figure:: ../images/surface_milo/iteration_3_embedding.webp

    UMAP embedding

  We can see that almost all patches have a membrane at the bottom now. Our task now becomes trying to eliminate patches by focusing on their centers. Specifically, we want to eliminate patches that have a carbon edge or gold particle or simply nothing at all, latter of which is more difficult.

  When we look at the cluster matrix, we see that there are a lot of clusters that seem to have bad patches but also some patches with spike proteins, which we would like to keep. As such, we have three options:

  #. Stop the iterations and continue with what we have.

  #. Continue iterations while still being filtering conservatively, at the cost of having to spend more time.

  #. Continue iterations but become less conservative at filtering, hoping that eliminating more false positives will outweigh the cost of eliminating true positives.

  We will do the third option to save time while also showing one more iteration.

  After checking all rows of the matrix, we decide to filter out these clusters::

    6, 18, 20, 22, 29, 35, 42, 47

.. nextpyp:: Iteration 4 (Training)
  :collapsible: closed

  In this iteration we decrease the patch size even further to focus on the spike proteins.

  * Click on ``MiLoPYP particles`` (output of the **third** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`MiLoPYP (train)`

  Use the same parameters as the previous block except:

  * On the **Pattern Mining** tab:

    - Set ``Epochs`` to 200

    - Set ``Bounding box (binned voxels)`` to 36

    - Set ``Learning rate`` to 0.01

    - Set ``Interval to perform validation`` to 20

    - Disable ``Remove Segmentation``

    - Set ``Patch coordinate location`` to the location of the ``/train/interactive_info_parquet.gzip`` file in the **third** **evaluation** block.

    - Set ``Class labels`` to ``6, 18, 20, 22, 29, 35, 42, 47``

    After we run the block, we can see that we further decreased our patch count from 1639 to 1368.

.. nextpyp:: Iteration 4 (Evaluation)
  :collapsible: closed

  * Click on ``MiLoPYP Model`` (output of the **fourth** :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

  * On the **Pattern Mining** tab:

    - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

    - Set ``Clusters`` to 60

  .. figure:: ../images/surface_milo/iteration_4_embedding.webp

    UMAP embedding

  We successfully eliminated all patches that were obviously bad. We can still see some patches with no obvious spike protein, but it's not easy to discriminate if they are actually background or just faint or noisy spikes. Nevertheless, we get to filter clusters once more before moving onto the refinement phase (without doing one more iteration), so we try to eliminate those we can.

  After checking all rows of the matrix, we decide to filter out these clusters::

    9, 28, 31, 32, 35, 36, 38, 44, 52, 56

.. nextpyp:: Exploration Phase Summary
  :collapsible: closed

  We did 4 iterations for the exploration phase in this tutorial. However, the strategies can be changed depending on the preferred trade off between time and accuracy. If the refinement phase is going to be used, then it is less problematic to delete some amount of true positives or keep small amount of false positives. If not, then it is likely better to be more careful during the iterations, and possibly change the parameters such as ``min/max distance`` or ``DoG sizes`` to start with more candidate locations.

  **Embeddings:**

  .. md-tab-set::

    .. md-tab-item:: No Surface Constraint

      .. figure:: ../images/surface_milo/no_surface_embedding.webp

    .. md-tab-item:: Iteration 1

      .. figure:: ../images/surface_milo/iteration_1_embedding.webp

    .. md-tab-item:: Iteration 2

      .. figure:: ../images/surface_milo/iteration_2_embedding.webp

    .. md-tab-item:: Iteration 3

      .. figure:: ../images/surface_milo/iteration_3_embedding.webp

    .. md-tab-item:: Iteration 4

      .. figure:: ../images/surface_milo/iteration_4_embedding.webp

  **Example Tomograms:**

  .. md-tab-set::

    .. md-tab-item:: No Surface Constraint

      .. figure:: ../images/surface_milo/no_surface_tomogram.webp

    .. md-tab-item:: Iteration 1

      .. figure:: ../images/surface_milo/iteration_1_tomogram.webp

    .. md-tab-item:: Iteration 2

      .. figure:: ../images/surface_milo/iteration_2_tomogram.webp

    .. md-tab-item:: Iteration 3

      .. figure:: ../images/surface_milo/iteration_3_tomogram.webp

    .. md-tab-item:: Iteration 4

      .. figure:: ../images/surface_milo/iteration_4_tomogram.webp

Refinement Phase
==========

Refinement phase also uses the segmentations to constrain particle picking to surfaces. This is done by creating a binary mask based on the distance from the segmentation. This binary mask is then used to filter input and output coordinates, in the loss function, or to modify the input tomograms to directly remove any signal outside the mask.

.. nextpyp:: Refinement (Training)
  :collapsible: closed

  * Click on ``MiLoPYP Particles`` (output of the **fourth** :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`Particle picking (train)`

  * On the **Training/Evaluation** tab:

    - Set ``Coordinates for training`` to *"class labels from MiLoPYP"*

      - Set ``Class IDs`` to ``0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 33, 34, 37, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 57, 58, 59``

    - Set ``Epochs`` to 3

    - Set ``Max number of particles`` to 600

    - Set ``Learning rate`` to 0.001

    - Set ``Validation interval (epochs)`` to 1

    - Enable ``Use masking``

      - Set ``Segmentation directory`` to the location of the ``/mrc`` folder in the **filtered** segmentation block (or **unfiltered** if your filters eliminate too many good components, you do not have many false positives from exploration phase, or you simply do not want to filter too much)

      - Set ``Mask radius`` to 25

    - Enable ``Enable Compilation``

      - Set ``Compile mode`` to *"max autotune"*

    - Enable ``Debug mode``

  After we run the block, we can see that we further decreased our patch count from 1368 to 1116.

  This block usually takes a while to run. However, it gives relatively good results even after a single epoch. You can increase or decrease the epoch count based on available time.

.. nextpyp:: Refinement (Evaluation)
  :collapsible: closed

  * Click on ``Particles Model`` (output of the :bdg-secondary:`Particle picking (train)` block) and select :bdg-primary:`Particle picking (eval)`

  Use the same parameters as the previous block except:

  * On the **Training/Evaluation** tab:

    - Set ``Trained model (*.pth)`` to the location of the model you want to evaluate

    - Set ``Particle radius (A)`` to 120

    - Set ``Threshold for soft/hard positives`` 0.85

Metrics
==========

.. nextpyp:: Representation Metrics
  :collapsible: closed

  * **mean_std_norm:**

    Average per-dimension standard deviation of L2-normalized embeddings. Very low values indicate possibly degenerate embeddings.

  * **pairwise_cosine:**

    Cosine similarities of randomly sampled embeddings. High values indicate that embeddings are close to each other.

  * **nearest_neighbor_cosine:**

    Cosine similarities of nearest neighbors of randomly sampled embeddings. Very high values indicate many near-identical embeddings.

  * **participation_ratio:**

    Effective dimensionality of the embeddings. Low values indicate most of the variability in the embeddings are concentrated in only a few directions.

.. nextpyp:: Clustering Metrics
  :collapsible: closed

  * **within_cos:**

    Average cosine similarity to the cluster centroid. Higher values indicate tighter clusters.

  * **between_cos:**

    Average cosine similarity between cluster centroids. Lower values indicate better seperated clusters.

  * **gap:**

    within_cos - between_cos, higher values generally indicate better clustering.

  * **davies_bouldin:**

    Davies–Bouldin index on the clusters. Lower values indicate better clustering.

  * **calinski_harabasz:**

    Calinski–Harabasz index on the clusters. Higher values indicate better clustering.

  * **centroid_participation_ratio:**

    Effective dimensionality of the cluster centroids. Low values indicate most of the variability in the clusters are concentrated in only a few directions.

.. nextpyp:: Stability Metrics
  :collapsible: closed

  * **ari_mean:**

    Average Adjusted Rand Index across repeated clusterings. Higher values indicate more consistent clustering.

  * **ari_std:**

    Standard deviation of Adjusted Rand Index across repeated clusterings. Lower values indicate more consistent clustering.
