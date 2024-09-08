===================
2D particle picking
===================

``nextPYP`` implements four types of methods for particle picking within the :badge:`Pre-processing,badge-secondary` block

Particle positions are stored as "lists", and multiple lists can be saved for a given dataset

Method 1: Import coordinates
============================

External coordinates can be imported as EMAN box files (``*.box``) or IMOD model files (``*.spk``):

#. Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab

#. Select "import" as the particle picking ``Method``
  
#. Specify the particle radius in A (for visualization purposes)

#. Select the location to ``Import particle coordinates (*.box, *.spk)`` (the folder should contain seaprate ``.box`` or ``.spk`` files for each micrograph)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to update the block

#. Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab to confirm that the particles were imported correctly

Method 2: Manual picking
========================

``nextPYP`` provides a convenient UI to pick particles manually:

#. Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab

#. Create a new list of particles by entering a name and clicking :badge:`New,badge-primary`

#. Select particles in the current micrograph by clicking on them

#. Navigate to other micrographs in the dataset usng the navigation bar and select more particles as needed

#. Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab
 
#. Select "manual" as the particle picking ``Method``
  
#. Specify the particle radius in A (for visualization purposes)

#. Choose the list of manually selected positions from the ``Select list for training`` dropdown menu

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to update the block

.. tip::

    - Particles can be deleted by right-clicking on the markers
    - Particle positions are saved automatically every time a particle is added or deleted
    - The total number of particles in a dataset is displayed on the top of the page
    - It is possible to :badge:`Copy,badge-primary`, :badge:`Delete,badge-primary`, and :badge:`Load,badge-primary` lists

Method 3: Size-based picking
============================

This method selects particle positions based on a target particle size:

#. Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab

#. Select "auto" or "all" as the particle picking ``Method`` ("auto" is more conservative, "all" tends to overpick)

#. Specify the particle radius in A and other parameters as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to update the block

#. Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab to inspect the results

Method 4: Neural-network picking
================================

Neural-network based methods require an existing list of particles for training a neural network. To pick particles, the trained model is then evaluated on the entire dataset. ``nextPYP`` uses a self-supervised approach that only needs sparsely annotated data. A wrapper for `Topaz <https://github.com/tbepler/topaz>`_ picking is also included. 

Training
^^^^^^^^

#. Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab
 
#. Select "pyp-train" or "topaz-train" as the particle picking ``Method``

#. Go to the corresponding **Training/Evaluation** tab and set the desired parameters

#. Choose a list of positions from the ``Select list for training`` dropdown menu

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to train the model

.. tip::
    
    - Since training requires a GPU, a GPU partition must be configured in the instance
    - The trained model(s) are saved in: ``train/YYYYMMDD_HHMMSS/*.training``
    - Challenging datasets may require the use of more particles for training

Evaluation
^^^^^^^^^^

#. Open the settings of the :badge:`Pre-processing,badge-secondary` block and go to the **Particle detection** tab
 
#. Select "pyp-eval" or "topaz-eval" as the particle picking ``Method`` (depending on which method was used for training)

#. Go to the corresponding **Training/Evaluation** tab and specify the location of the trained model (``*.training``)

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary` to pick particles on all micrographs

#. Navigate to the :badge:`Pre-processing,badge-secondary` block and go to the **Micrographs** tab to inspect the results

.. seealso::

    * :doc:`3D particle picking<picking3d>`
    * :doc:`Pattern mining (MiLoPYP)<milopyp>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`