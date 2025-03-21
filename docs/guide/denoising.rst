==================
Tomogram denoising
==================

``nextPYP`` provides wrappers for several tomogram denoising methods based on neural networks, including `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_, `Topaz-Denoise <https://github.com/tbepler/topaz>`_ and `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_

Denoising is done in two phases using the :bdg-secondary:`Denosing (training)` and :bdg-secondary:`Denosing (eval)` blocks. If a pre-trained model is available, the training phase can be skipped. Training is only supported for cryoCARE and IsoNet models (Topaz-denoise uses pre-trained models)

Training
~~~~~~~~

* Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Denoising (train)`

* Select the desired denoising method and corresponding parameters for training

* (optional) To train models on a subset of tomograms, create a :doc:`Filter<filters>` in the :bdg-secondary:`Pre-processing` block and select its name from the `Filter tomograms` dropdown menu

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-secondary:`Denoising (training)` block to inspect the results of training

.. note::

    This step can be skipped if a pre-trained model is available

Evaluation
~~~~~~~~~~

* Click on ``Denoising model`` (output of the :bdg-secondary:`Denoising (traiing)` block) and select :bdg-primary:`Denoising (eval)`

* Select the desired algorithm and corresponding trained model from the block upstream (cryoCARE and IsoNet) or list of pre-trained models (Topaz)

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Navigate to the :bdg-secondary:`Denoising (eval)` block to inspect the denoised tomograms


.. admonition:: Notes

    * Evaluation is typically done on the entire set of tomograms, while training is done using a subset of tomograms
    * cryoCARE and IsoNet need a GPU to run. Topaz can run on the CPU (default) or a GPU

Denoised tomograms can be used as input for all downstream operations available in ``nextPYP`` (e.g., particle picking, segmentation, etc.)

.. admonition:: Note

    All downstream particle refinement operations will still use the raw data (regardless of whether denoising was used for particle picking or not)
