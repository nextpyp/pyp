==================
Tomogram denoising
==================

``nextPYP`` provides wrappers for several tomogram denoising methods, including `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_, `Topaz-Denoise <https://github.com/tbepler/topaz>`_ and `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_.


One-step denosing (Topaz)
-------------------------

One-step denoising is done in the :bdg-secondary:`Denosing` block using Topaz-denoise (which uses a pre-trained model).

* Click on :guilabel:`Tomograms` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Denoising`

* Select the desired algorithm and corresponding parameters

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-primary:`Reconstruction` tab to inspect the denoised tomograms


Two-step denosing (cryoCARE, IsoNet)
------------------------------------

Two-step denoising is done in the :bdg-secondary:`Denosing (training)` and :bdg-secondary:`Denosing (eval)` blocks using cryoCARE or IsoNet.

Training
~~~~~~~~

* Click on :guilabel:`Tomograms` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Denoising (train)`

* Select the desired method and corresponding parameters

* (optional) To train on a subset of tomograms, create a :doc:`Filter<filters>` in the :bdg-secondary:`Pre-processing` block and select its name from the `Filter tomograms` dropdown menu

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-primary:`Denoising (training)` block to inspect the results of training


Evaluation
~~~~~~~~~~

* Click on :guilabel:`Denoising model` (output of the :bdg-secondary:`Denoising (traiing)` block) and select :bdg-primary:`Denoising (eval)`

* Select the algorithm and trained model from the block upstream

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-primary:`Denoising (eval)` block to inspect the denoised tomograms


.. note::

    Evaluation is always done on the entire set of tomograms from the pre-processing block
