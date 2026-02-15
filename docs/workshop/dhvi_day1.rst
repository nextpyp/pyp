#####################
DHVI workshop (day 1)
#####################

On this session, we will import raw data from two datasets from the EMPIAR database:
#. Immature Gag protein from HIV-1 VLPs (`EMPIAR-10164 <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_)
#. Spike protein from SARS-CoV-2 virions (`EMPIAR-10453 <https://www.ebi.ac.uk/empiar/EMPIAR-10453/>`_)

Session 1 - Data import and pre-processing
==========================================

For this session, we will use the HIV-1 dataset that was collected on a 300kV Titan Krios TEM using a Gatan K2 detector at 1.35A per pixel. Each tilt-series has 41 images recorded between -60 and 60 degrees with a spacing of 3 degrees. Each tilt image was collected as a sequence of 8-10 frames.

.. nextpyp:: Step 1: Create a project
  :collapsible: open
  
  Data processing runs are organized into projects. We will create a new project for this tutorial:

  * Click on :bdg-primary:`Create new project`, give the project a name, and select :bdg-primary:`Create`

  * Select the new project from the **Dashboard** and click :bdg-primary:`Open`

  * The newly created project will be empty and a **Jobs** panel will appear on the right

.. nextpyp:: Step 2: Import raw tilt-series
  :collapsible: open

  * Go to :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`  

  * Go to the **Raw data** tab:

    - Set the ``Location`` of the raw data by clicking on the :fa:`search` icon and browse to ``/nfs/bartesaghilab/nextpyp/workshop_dhvi/10164/``
    
    - Type ``*.tif`` into the filter box (lower right) and click the :fa:`filter` icon
       
  * Go the the **Microscope Parameters** tab: 

    - Set ``Pixel size (A)`` to 1.35

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 85.3    

  * Click :bdg-primary:`Save` and the new block will appear on the project page. The block is in the modified state (indicated by the :fa:`asterisk` sign) and is ready to be executed

  * Clicking the :bdg-primary:`Run` button will show another dialog where you can select which blocks to run:

  * Click :bdg-primary:`Start Run for 1 block`. This will launch a process that reads one tilt at random and displays the resulting image inside the block

  * Click on the thumbnail inside the block to see a larger version of the result

.. nextpyp:: Step 3: Pre-processing
  :collapsible: open
  
  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

  * Go to the **Frame alignment** tab:

    - ``nextPYP`` uses the ``Frame pattern`` to extract metadata form the file names. EMPIAR-10164 follows the default file naming scheme and ``.tif`` extension, so we will leave the default setting. 

    - We will use ``unblur`` for frame alignment. 

  * Go to the **Tilt-series alignment** tab

    - Our ``Alignment method`` will be IMOD fiducial-based which is the default so make no changes.
  
  * Go to the **CTF determination** tab

    - Set ``Max resolution`` to 5 

  * Go to the **Tomogram reconstruction** tab
  
    - Our ``Reconstruction method`` will be IMOD, this is the default so make no changes. 

  * Go to the **Resources** tab

    - Set ``Split, Threads`` to 11
  
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

  * When the block finishes running, examine the **Tilt-series**, **Plots**, **Table**, and **Gallery** tabs.  


Session 2 - Tomogram denoising and segmentation
===============================================

For this session, we will use the SARC-VoV-2 dataset that was collected on a 300kV Titan Krios TEM using a Gatan K2 detector at 1.329A per pixel. Each tilt-series has 41 images recorded between -60 and 60 degrees with a spacing of 3 degrees. No frames are available for this dataset.

Import data and pre-processing
------------------------------

.. nextpyp:: Step 1: Import raw tilt-series
  :collapsible: open

  * Click :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`

  * On the **Raw data** tab:

    - Set the ``Location`` by clicking on the :fa:`search` icon and browsing to ``/nfs/bartesaghilab/nextpyp/workshop_dhvi/10453/``

    - Type ``*.mrc`` into the filter box (lower right) and click the :fa:`filter` icon

  * On the **Microscope parameters** tab:

    - Set ``Pixel size (A)`` to 1.329

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 84.8

  * Click :bdg-primary:`Save` and :bdg-primary:`Run` the block.

.. nextpyp:: Step 2: Pre-processing
  :collapsible: open

  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

  * On the **Frame alignment** tab:

    - Enable ``No movie frames``

  * On the **Tilt-series alignment** tab:

    - Disable ``Reshape tilt-images into squares``

  * On the  **CTF determination** tab:

    - Set ``Max resolution (A)`` to 10

  * On the **Tomogram reconstruction** tab:

    - Set ``Tomogram thickness (unbinned voxels)`` to 1536

    - Set ``2D filtering`` to *"none"*

    - Enable ``Erase fiducials``

    - Enable ``Generate half-tomograms``

    - Set ``High-frequency filtering`` to *"hamming (as in tomo3d)"*

  * On the **Resources** tab

    - Set ``Split, Threads`` to 11

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

Tomogram segmentation
---------------------

For segmenting tomograms, we will use pre-trained `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_ models that can be directly applied to tomograms produced by the pre-processing block. In addition, we will clean up the segmentaion results using different geometric criteria.

.. nextpyp:: Step 3: Segmentation
  :collapsible: open

  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Segmentation (membrain/tardis)`

  * On the  **Tomogram segmentation** tab:

    - Enable ``Pre-process tomograms``

    - Set ``Pixel size rescaling`` to 11

    - Enable ``Deconvolution filter``

    - Set ``Pre-trained model (*.ckpt)`` to ``/nfs/bartesaghilab/nextpyp/workshop_dhvi/models/MemBrain_seg_v10_alpha.ckpt``

    - Set ``Filter connected components`` to *"by number"*

    - Set ``Components to keep`` to 10

    - Set ``Thickness of slab to keep (unbinned voxels)`` to 1228

    - Enable ``Test time augmentation``

    - Set ``Sliding window size`` to 96

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

nextPYP can also use the `TARDIS <https://github.com/SMLC-NYSBC/TARDIS>`_ package for tomogram segmentation. The workflow for running TARDIS is identical to MemBrain-Seg's, with the exception that some of the parameters are different.

Tomogram denoising
------------------

For denoising tomograms, we will demonstrate the use of `IsoNet2 <https://github.com/IsoNet-cryoET/IsoNet2>`_. Unlike segmentation models that can generalize well to different type os data, denoising models typically require re-training to achieve the best performance.

.. nextpyp:: Step 4: Denoising (training)
  :collapsible: open

  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Denoising (train)`

  * On the  **Tomogram denoising** tab:

    - Set ``Method`` to *"isonet2"*

    - Enable ``Use masking``

      - Set ``Epochs for training`` to 30

      - Set ``Loss function`` to *"Huber"*

      - Set ``Learning rate`` to 0.0001

      - Set ``Minimum learning rate`` to 0.0001

      - Set ``B-factor`` to 200

    - Set ``Epochs for training`` to 30

    - Set ``Loss function`` to *"Huber"*

    - Set ``Learning rate`` to 0.0001

    - Set ``Minimum learning rate`` to 0.0001

    - Set ``Missing wedge weight in loss`` to 100

    - Set ``CTF mode`` to *"network"*

    - Set ``B-factor`` to 200

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

.. nextpyp:: Step 5: Denoising (evaluation)
  :collapsible: open

  * Click on ``Denoising model`` (output of the :bdg-secondary:`Denoising (train)` block) and select :bdg-primary:`Denoising (eval)`

  * On the  **Tomogram denoising** tab:

    - Set ``Method`` to *"isonet2"*

    - Set ``Trained model`` to *"/nfs/bartesaghilab/nextpyp/workshop_dhvi/models/isonet_network_isonet2-n2n_unet-medium_96_epoch30_full.pt"*

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

nextPYP also supports other algorithms for denoising, including `Topaz-Denoise <https://github.com/tbepler/topaz>`_, `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_, `Noise2Map <https://warpem.github.io/warp/reference/noise2map/noise2map/?h=noise>`_, and `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_. The workflow for these algorithms uses the same sequence of steps we saw here.

Day 1 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  * How to import raw data

  * How to pre-process tilt-series (frame alignment, tilt-series alignment, tilted CTF estimation)

  * How to reconstruct tomograms (set binning and thickness, erase fiducials)

  * How to segment tomograms using `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_

  * How to train and evaluate `IsoNet2 <https://github.com/IsoNet-cryoET/IsoNet>`_ models to denoise tomograms

  :doc:`On day 2<dhvi_day2>` we will demonstrate ``nextPYP``'s functionality for particle picking.