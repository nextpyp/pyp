===================
2D particle picking
===================

``nextPYP`` implements multiple methods for particle picking as part of the :badge:`Pre-processing,badge-secondary` block

Particle positions are stored as "lists", and each pre-processing can have multiple lists

You can create a new list by entering a name and clicking :badge:`New,badge-primary`. It is also possible to copy, delete, or load previously saved lists using the :badge:`Copy,badge-primary`, :badge:`Delete,badge-primary`, and :badge:`Load,badge-primary` buttons

Method 1: Import coordinates
============================

External coordinates can be imported from EMAN box files (``*.box``) or IMOD model files (``*.spk``)

- Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab

- Select "import" as ``Method``
  
- Specify the particle radius in A (for visualization purposes)

- Specify the location to ``Import particle coordinates (*.box, *.spk)`` (the folder should contain one ``.box`` or ``.spk`` file for each micrograph in the dataset)

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to update the block

- Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab to confirm that the particles were imported correctly

Method 2: Manual picking
========================

- Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab

- Create a new list by entering a name and clicking :badge:`New,badge-primary`

- Select particles in the current micrograph by clicking on them

- Navigate to other micrographs in the dataset usng the navigation bar and select particles as needed

- Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab
 
- Select "manual" as ``Method``
  
- Specify the particle radius in A (for visualization purposes)

- Choose the list of manually selected positions from the ``Select list for training`` dropdown menu at the top of the form

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to update the block

.. note::

    - Particles can be deleted by right-clicking on the markers
    - Particle positions are saved automatically every time a particle is added or deleted
    - The total number of particles in a dataset is displayed on the top of the page

.. figure:: ../images/guide_nn_picking_2d.webp
    :alt: Create new filter

Method 3: Size-based picking
============================

- Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab

- Select "auto" or "all" as ``Method`` to detect particles based on size. "auto" is more conservative and gives fewer particles while "all" gives more particles

- Specify the particle radius in A and other parameters as needed

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to update the block

- Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab to inspect the results

Method 4: Neural-network picking
================================

``nextPYP`` provides a network for joint denoising-picking  ("pyp-train"/"pyp-eval") and wrappers for `Topaz <https://github.com/tbepler/topaz>`_ ("topaz-train"/"topaz-eval")

These methods require an existing list of particles and have two stages

#. Training
^^^^^^^^^^^

- Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab
 
- Select "pyp-train" or "topaz-train" as ``Method``

- Go to the corresponding **Training/Evaluation** tab and set the desired parameters

- Choose a list of positions from the ``Select list for training`` dropdown menu at the top of the form

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to train the model

.. note::
    
    - Since training is run on the GPU, a GPU partition must be configured in the ``nextPYP`` instance
    - The model(s) will be saved as ``train/YYYYMMDD_HHMMSS/*.training``
    - Since the quality of the picking may depend on the size of the training set, challenging datasets may require the use of more particles for training

#. Evaluation
^^^^^^^^^^^^^

- Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab
 
- Select "pyp-eval" or "topaz-eval" as ``Method`` (depending on which method was used for training)

- Go to the corresponding **Training/Evaluation** tab and specify the location of the trained model (``*.training``)

- Click :badge:`Save,badge-primary`, then :badge:`Run,badge-primary` to evaluate the model on all the micrographs

- Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab to inspect the results

.. seealso::

    * :doc:`Particle picking<picking>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`