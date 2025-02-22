===================
2D Particle picking
===================

``nextPYP`` implements four methods for particle picking within the :bdg-secondary:`Pre-processing` block

Particle positions are stored as "lists". Multiple lists can be saved for each :bdg-secondary:`Pre-processing` block

Import coordinates
==================

External coordinates can be imported as EMAN box files (``*.box``) or IMOD model files (``*.spk``):

#. Open the settings of the :bdg-secondary:`Pre-processing` block and go to the **Particle detection** tab

#. Select "import" as the particle picking ``Method``
  
#. Specify the particle radius in A (for visualization purposes)

#. Select the location to ``Import particle coordinates (*.box, *.spk)`` (the folder should contain separate ``.box`` or ``.spk`` files for each micrograph)

#. Click :bdg-primary:`Save,`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to update the block

#. Navigate to the :bdg-secondary:`Pre-processing` block and go to the **Micrographs** tab to confirm that the particles were imported correctly

Manual picking
==============

``nextPYP`` provides a convenient UI to pick particles manually:

#. Navigate to the :bdg-secondary:`Pre-processing` block and go to the **Micrographs** tab

#. Create a new list of particles by entering a name and clicking :bdg-primary:`New`

#. Select particles in the current micrograph by simply clicking on them

#. Navigate to other micrographs in the dataset using the navigation bar and select more particles as needed

#. Open the settings of the :bdg-secondary:`Pre-processing` block and go to the **Particle detection** tab
 
#. Select "manual" as the particle picking ``Method``
  
#. Specify the particle radius in A (for visualization purposes)

#. Choose the list of manually selected positions from the ``Select list for training`` dropdown menu

#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to update the block

.. tip::

    - Particles can be deleted by right-clicking on the green markers
    - Particle positions are saved automatically every time a particle is added or deleted
    - The total number of particles in a dataset is displayed at the top of the page
    - It is possible to :bdg-primary:`Copy`, :bdg-primary:`Delete`, and :bdg-primary:`Load` lists

Size-based picking
==================

This method selects particles based on their size:

#. Open the settings of the :bdg-secondary:`Pre-processing` block and go to the **Particle detection** tab

#. Select "auto" or "all" as the particle picking ``Method`` ("auto" is more conservative, "all" tends to overpick)

#. Specify the particle radius in A and other parameters as needed

#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to update the block

#. Navigate to the :bdg-secondary:`Pre-processing` block and go to the **Micrographs** tab to inspect the results

Neural-network picking
======================

Neural-network based methods require an existing list of particles for training a model. To pick particles, the trained model is then evaluated on the entire dataset. ``nextPYP`` uses a self-supervised approach that only needs sparsely annotated data. A wrapper for `Topaz <https://github.com/tbepler/topaz>`_ picking is also included. 

Training
^^^^^^^^

#. Open the settings of the :bdg-secondary:`Pre-processing` block and go to the **Particle detection** tab
 
#. Select "pyp-train" or "topaz-train" as the particle picking ``Method``

#. Go to the corresponding **Training/Evaluation** tab and set the desired parameters

#. Choose a list of positions from the ``Select list for training`` dropdown menu

#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to train the model

.. tip::
    
    - Since training runs on the GPU, a standalone GPU-server is required (or GPU partitions must be properly configured in SLURM)
    - The trained model(s) are saved in the project directory under: ``train/YYYYMMDD_HHMMSS/*.training``
    - Challenging datasets may require the use of more particles for training

Evaluation
^^^^^^^^^^

#. Open the settings of the :bdg-secondary:`Pre-processing` block and go to the **Particle detection** tab
 
#. Select "pyp-eval" or "topaz-eval" as the particle picking ``Method`` (depending on which method was used for training)

#. Go to the corresponding **Training/Evaluation** tab and specify the location of the trained model (``*.training`` file)

#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block` to pick particles on all micrographs

#. Navigate to the :bdg-secondary:`Pre-processing` block and go to the **Micrographs** tab to inspect the results
