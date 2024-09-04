==================
Tomogram denoising
==================

``nextPYP`` has wrappers for several tomogram denosing methods, including `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_, `Topaz-Denoise <https://github.com/tbepler/topaz>`_ and `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_.


One-step denosing (cryoCARE, Topaz)
-----------------------------------

One-step denoising is done in the :badge:`Denosing,badge-secondary` block using cryoCARE (which needs to be trained on each pair of half-tomograms) or Topaz-denoise (which uses a pre-trained model).

* Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Denoising,badge-primary`

* Select the desired algorithm and corresponding parameters

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Reconstruction,badge-primary` tab to inspect the denoised tomograms


Two-step denosing (IsoNet)
--------------------------

Two-step denoising is done in the :badge:`Denosing (training),badge-secondary` and :badge:`Denosing (eval),badge-secondary` blocks using IsoNet.

1. Training
~~~~~~~~~~~

* Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`Denoising (train),badge-primary`

* Select the desired algorithm and corresponding parameters

* (optional) To train on a subset of tomograms, create a :doc:`Filter <filters>`_ in the :badge:`Pre-processing,badge-secondary` block and select its name from the `Filter tomograms` dropdown menu

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Denoising (training),badge-primary` block to inspect the results of training


2. Evaluation
~~~~~~~~~~~~~

* Click on :guilabel:`Denoising model` (output of the :badge:`Denoising (traiing),badge-secondary` block) and select :badge:`Denoising (eval),badge-primary`

* Select the algorithm and trained model from the block upstream

* Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

* Navigate to the :badge:`Denoising (eval),badge-primary` block to inspect the denoised tomograms


.. note::

    Evalaution is always done on the entire set of tomograms from the pre-processing block


.. seealso::

    * :doc:`Particle picking<picking>`
    * :doc:`Segmentation<segmentation>`
    * :doc:`Pattern mining<milopyp>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`