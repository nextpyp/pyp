=====================================
Neural-network based particle picking
=====================================

``nextPYP`` implements semi-supervised particle picking using neural-networks both for 2D micrographs and 3D tomograms

Step 1: Pick particles for training
-----------------------------------

.. tabbed:: 2D picking

  - Click inside the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab

  - Create a new list by entering a name and clicking :badge:`New,badge-primary`

  - Select particles in the current micrograph by clicking on their centers

  - Navigate to other micrographs in the dataset and select additional particles as needed

  .. figure:: ../images/guide_nn_picking_2d.webp
      :alt: Create new filter

.. tabbed:: 3D picking

  - Click inside the :badge:`Pre-processing,badge-secondary` block, go to the **Tilt-series** tab, and select the **Reconstruction** group

  - Create a new list by entering a name and clicking :badge:`New,badge-primary`

  - Select particles in the current tomogram by clicking on their centers. Use the slider below the image to scroll through the tomogram

  - Navigate to other tomograms in the dataset and select additional positions as needed

  .. figure:: ../images/guide_nn_picking_3d.webp
      :alt: Create new filter

.. note::

    - Particles can be deleted by right-clicking on the markers
    - Particle positions are saved automatically every time a particle is added or deleted
    - The total number of particles in a dataset is displayed on the top-left corner of the page

.. tip::

    The size of the markers can be controlled by changing the ``Detection radius`` in the **Particle detection** tab. The block must be re-run for this change to take effect

Step 2: Train the neural-network model
--------------------------------------

- Open the settings of the :badge:`Pre-processing,badge-secondary` block, go to the **Particle detection** tab and select `nn-train` as the ``Detection method``

- Choose the list of manually selected positions from the ``Select list for training`` dropdown menu at the top of the form

.. figure:: ../images/guide_nn_picking_select_list.webp
    :alt: Create new filter

- Go to the **Training/Evaluation** tab and set the desired parameters for training

.. figure:: ../images/guide_nn_picking_select_params.webp
    :alt: Create new filter

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to train the model

.. note::

    Since training is run using the GPU, a GPU partition must be configured in the nextPYP instance

Step 3: Run inference using the trained model
---------------------------------------------

- Go to the **Particle detection** tab in the :badge:`Pre-processing,badge-secondary` block and select `nn-eval` as the ``Detection method``

- Go to the **Training/Evaluation** tab and select the location of the trained model obtained in the previous step (``train/YYYYMMDD_HHMMSS/*.training`` for 2D, and ``train/YYYYMMDD_HHMMSS/*.pth`` for 3D)

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to evaluate the model on all the micrographs or tomograms

- Inspect the results using the **Micrographs** tab (2D) or the **Reconstruction** group in the **Tilt-series** tab (3D)

.. tip::

    Since the quality of the picking may depend on the size of the training set, challenging datasets may require the use of more particles for training

.. seealso::

    * :doc:`Particle picking<picking>`
    * :doc:`Filters<filters>`
    * :doc:`Overview<overview>`