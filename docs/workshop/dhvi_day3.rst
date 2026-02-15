###################
DHVI course (day 3)
###################

Session 1: Sub-tomogram averaging
==================================

Traditional sub-tomogram alignment-through-classification is a powerful strategy for *de novo* structure determination. It involves iterative 3D classification, alignment, and averaging of sub-volumes. Initially, homogeneous particle groups are identified through 3D classification and subsequently averaged in 3D. The resulting class averages are then aligned to one another and combined into high signal-to-noise (SNR) references, which can be used to align individual sub-volumes. The resulting 3D models can then serve as references for high-resolution refinements using 2D projections.


.. nextpyp:: Step 1: Sub-volume generation
  :collapsible: open

  #. To access sub-volumes for averaging, generate them via the :bdg-secondary:`Particle picking` block. 
  
  * On the **Particle extraction** tab: 
  
    - Set ``Sub-volume export format`` to *"3davg"*
    
    - Set ``Sub-tomogram size (voxels)`` to 64
    
    - Set ``Sub-tomogram binning`` to 2

    - Disable ``Use existing reconstruction settings``

    - Set ``2D filtering`` to *"none"*

    - Enable ``Erase fiducials``

    - Set ``Radial filtering`` to *"fakeSIRT (mimic SIRT reconstruction)"*

    - Set ``Number of iterations`` to 5

  #. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.


.. nextpyp:: Step 2: Global average and centering
  :collapsible: open

  After generating the sub-volumes, crate and configure the :bdg-secondary:`Sub-tomogram averaging` block to run Mode 0:

  #. Click on ``Particles`` (output of the :bdg-secondary:`Particle picking` block), then choose :bdg-primary:`Sub-tomogram averaging`. This will create a new block and show the form to enter parameters.
  #. Under ``Alignments from sub-volume averaging (*.txt)``, navigate to the ``frealign`` directory from the upstream block and select the ``*_original_volumes.txt`` file.
  #. Choose `mode 0 - global average and centering` as the ``Refinement mode``.
  #. To use a radially symmetrized average for centering, enable ``Rotational symmetry``, set the number of centering iterations, and adjust any masking or filtering settings.
  #. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
  #. Review results in the **Global averages** tab within the :bdg-secondary:`Sub-tomogram averaging` block.

.. nextpyp:: Step 3: 3D classification
  :collapsible: open

  To perform 3D classification (Mode 1):

  #. Return to the project page and select ``Edit`` from the block menu.
  #. Choose `mode 1 - classification` as the ``Refinement mode``.
  #. Set the number of desired classes and configure masking or filtering as needed.
  #. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
  #. View results in the **Classes** tab of the :bdg-secondary:`Sub-tomogram averaging` block.

.. nextpyp:: Step 4: Selection and alignment of class averages
  :collapsible: open

  To align selected class averages to a reference (Mode 2):

  #. Return to the project page and select ``Edit`` from the block menu.
  #. Choose `mode 2 - alignment of averages` as the ``Refinement mode``.
  #. Specify the class selection, listing the reference class first (e.g., `5,1,3,4` aligns classes 1, 3, and 4 to class 5).
  #. Set masking and filtering options as needed.
  #. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
  #. Review aligned classes in the **Classes (aligned)** tab of the :bdg-secondary:`Sub-tomogram averaging` block.

.. nextpyp:: Step 5: Alignment of sub-tomograms to reference
  :collapsible: open

  To align all sub-volumes to the generated reference (Mode 3):

  #. Return to the project page and select ``Edit`` from the block menu.
  #. Choose `mode 3 - alignment to reference` as the ``Refinement mode``.
  #. Set the parameters for rotational and translational search, along with masking and filtering options.
  #. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

.. nextpyp:: Step 6: Iterative refinement
  :collapsible: open

  #. Return to the project page and select ``Edit`` from the block menu.
  #. Increase the ``Iteration number`` to 2 and repeat steps 2-4 (in that order) to iteratively refine your model.

Session 2: Constrained single-particle tomography
=================================================

Now, we will import 19,972 HIV-Gag protein particles, import initial reference-based alignments, then go through a condensed version of the 3D Refinement pipeline to attain an ~4Å resolution structure from 5,000 filtered particles. At a high level, we will be performing reference-based refinement, filtering particles, performing region-based refinement and tilt-geometry refinement. 

.. nextpyp:: Step 1: Import particles
  :collapsible: open
  
  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Set ``Detection method`` to *"import"*

  * Set ``Particle radius (A)`` to 50 

  * Click :fa:`search` and browse to ``/nfs/bartesaghilab/nextpyp/workshop_dhvi/10164/particles``. Select :bdg-primary:`Choose Folder`

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step 2: Import alignments
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Pickng` block) and select :bdg-primary:`Calculate reconstruction`

  * Go to the **Sample** tab 
    
    - Set ``Molecular weight (kDa)`` to 300 

    - Set ``Particle radius (A)`` to 150 

    - Set ``Symmetry`` to C6

  * Go to the **Extraction** tab

    - Set ``Box size (pixels/voxels)`` to 128 

    - Set ``Image binning`` to 2

  * Go to the **Alignments** tab

    - From the ``Import from`` dropdown menu, select ``nextPYP (*.bz2)``

    - Click the :fa:`search` icon next to ``Input parameter file (*.bz2)`` and browse to ``/nfs/bartesaghilab/nextpyp/workshop_dhvi/10164/tomo-coarse-refinement-fg2v2MJLSY4Ui908_r01_02.bz2``  Click :bdg-primary:`Choose File`

  * Go to the **Reconstruction** tab

    - Select ``Apply dose weighting`` by checking the box 

  * Go to the **Resources** tab

    - Set ``Split, Threads`` to 124

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/cspt.webp
      
      Constrained single-particle tomography (CSPT)

.. nextpyp:: Step 3: Particle filtering
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`Particle filtering`

  * Go to the **Particle filtering** tab

    - Set ``Score threshold`` to 3.5

    - Set ``Min distance between particles (unbinned pixels)`` to 54

    - Click the :fa:`search` icon next to ``Input parameter file(*.bz2)`` and select the ``*.bz2`` file that appears (this is from the parent directory). Click :bdg-primary:`Choose File`

    - Check the box next to ``Permanently remove particles``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step 4: Region-based refinement, tilt-geometry refinement, further particle refinement
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle filtering` block) and select :bdg-primary:`3D refinement`

  * Go to the **Extraction** tab

    - Set ``Box size (pixels/voxels)`` to 256

    - Set ``Image binning`` to 1

  * Go to the **Particle scoring function** tab

    - Set ``Last tilt for refinement`` to 8 

    - Set ``Max resolution (A)`` to 4:3.5

    - From the ``Masking strategy`` dropdown menu, select ``from file``

    - Click the :fa:`search` icon to select the ``Shape mask (*.mrc)``, browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc``, and click :bdg-primary:`Choose File`

  * Go to the **Refinement** tab

    - Next to ``Input parameter file (*.bz2)`` click the :fa:`search` icon. Select the ``_r01_02_clean.bz2`` file and click :bdg-primary:`Choose File`

    - Set ``Last iteration`` to 3

    - Check ``Refine tilt-geometry``

    - Check ``Refine particle alignments`` 

    - Set ``Number of regions`` to 8,8,2 

  * Go to the **Reconstruction** tab

    - Check ``Apply dose weighting`` (It may already be checked)

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/regionbased.webp
      
      Region-based refinement



Day 3 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  In this session, we learned about the following topics in 3D refinement:
  
  * Sub-volume averaging and classification

  * Import particle coordinates and alignments

  * Constrained single-particle tomography (CSPT)

  * Map post-processing

  * Visualization of results in ChimeraX and ArtiaX

  ``nextPYP`` also supports particle-based CTF refinement, building shape masks, ab-initio refinement, and 3D classification

  :doc:`On day 4<dhvi_day4>` we will demonstrate ``nextPYP``'s functionality for post-processing and on-the-fly data processing.