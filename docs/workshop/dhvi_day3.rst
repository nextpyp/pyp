#####################
DHVI workshop (day 3)
#####################

Session 1: Sub-tomogram averaging
==================================

Traditional sub-tomogram averaging is a powerful strategy for *de novo* structure determination that involves doing iterative 3D classification, alignment, and averaging of sub-tomograms. The results can be used as a starting point for high-resolution refinement using constrained single-particle tomography.

For this session, we will use the particles we obtained yesterday for the **SARS-CoV-2** dataset.

.. nextpyp:: Step 1: Sub-volume extraction
  :collapsible: open

  * We first need to generate the sub-tomograms via the :bdg-secondary:`Particle picking (eval)` block. 
  
  * On the **Particle extraction** tab:
  
    - Set ``Sub-volume export format`` to *3davg (do not invert contrast)*
    
    - Set ``Sub-tomogram size (voxels)`` to 64
    
    - Set ``Sub-tomogram binning`` to 4

    - Disable ``Use existing reconstruction settings``

      - Set ``2D filtering`` to *none*

      - Enable ``Erase fiducials``

      - Set ``Radial filtering`` to *fakeSIRT (mimic SIRT reconstruction)*

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

After generating the sub-volumes, we will create and configure the :bdg-secondary:`Sub-tomogram averaging` block by clicking on ``Particles`` (output of the :bdg-secondary:`Particle picking (eval)` block), then choosing :bdg-primary:`Sub-tomogram averaging`

The sub-volume averaging pipeline in ``nextPYP`` consists of an iterative procedure that alternates between four runnning modes:

#. **Mode 0** - Global averaging and iterative centering

#. **Mode 1** - 3D classification

#. **Mode 2** - Alignment of class averages

#. **Mode 3** - Alignment of raw sub-volumes to reference

.. nextpyp:: Iteration 1
  :collapsible: open

  .. md-tab-set::

    .. md-tab-item:: Mode 0

      * On the **Sub-volume averaging** tab

        * Set ``Alignments from sub-volume averaging (*.txt)`` by navigating to the ``frealign`` directory from the upstream block and selecting the file ``*_original_volumes.txt``
        * Set ``Centering iterations`` to 1
        * Set ``Translation tolerance (binned voxels)`` to 20
        * Set ``Mask radius (binned voxels)`` to 24,24,26
        * Set ``Mask apodization (binned voxels)`` to 0
        * Set ``Low-pass filter (cutoff, decay)`` to 0.2,0.05
        * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
      
      * Review results in the **Centering** tab within the :bdg-secondary:`Sub-tomogram averaging` block.

    .. md-tab-item:: Mode 1

      * On the **Sub-volume averaging** tab

        * Set ``Refinement mode`` to *mode 1 - classification*
        * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
      
      * Review the results in the **Classification** tab of the :bdg-secondary:`Sub-tomogram averaging` block.

    .. md-tab-item:: Mode 2

      * On the **Sub-volume averaging** tab

        * Set ``C symmetry order`` to 12
        * Set ``Refinement mode`` to *mode 2 - alignment of averages*
        * Set ``Class selection`` to a comma separated list of good classes, listing the reference class first (e.g., `5,1,3,4` aligns classes 1, 3, and 4 to class 5).
        * Set ``Out-of-plane rotation tolerance (degrees)`` to 30
        * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
        
      * Review aligned classes in the **Aligned classes** tab of the :bdg-secondary:`Sub-tomogram averaging` block.

    .. md-tab-item:: Mode 3

      * On the **Sub-volume averaging** tab

        * Set ``C symmetry order`` to 1
        * Set ``Refinement mode`` to *mode 3 - alignment to reference*
        * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
        
      * On the **Resources** tab

        - Set ``Split, Threads`` to 124

      * Review the results in the **References** tab of the :bdg-secondary:`Sub-tomogram averaging` block.

.. nextpyp:: Iteration 2
  :collapsible: closed

  * Set ``Iteration number`` to 2

  .. md-tab-set::

    .. md-tab-item:: Mode 1

      * Set ``Refinement mode`` to *mode 1 - classification*
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. md-tab-item:: Mode 2

      * Set ``Refinement mode`` to mode 2 - alignment of averages*
      * Set ``Class selection`` to a comma separated list of good classes, listing the reference class first
      * Set ``Out-of-plane rotation tolerance (degrees)`` to 10
      * Set ``Translation tolerance (binned voxels)`` to 10
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
      
    .. md-tab-item:: Mode 3

      * Set ``Refinement mode`` to *mode 3 - alignment to reference*
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.      

.. nextpyp:: Iteration 3
  :collapsible: closed

  * Set ``Iteration number`` to 3

  .. md-tab-set::

    .. md-tab-item:: Mode 1

      * Set ``Refinement mode`` to *mode 1 - classification*
      * Set ``Mask radius (binned voxels)`` to 22,22,26
      * Set ``Low-pass filter (cutoff, decay)`` to 0.25,0.05
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

    .. md-tab-item:: Mode 2

      * Set ``Refinement mode`` to *mode 2 - alignment of averages*
      * Set ``Class selection`` to a comma separated list of good classes, listing the reference class first
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

    .. md-tab-item:: Mode 3

      * Set ``Refinement mode`` to *mode 3 - alignment to reference*
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

.. nextpyp:: Iteration 4
  :collapsible: closed

  * Set ``Iteration number`` to 4

  .. md-tab-set::

    .. md-tab-item:: Mode 1

      * Set ``Refinement mode`` to *mode 1 - classification*
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

    .. md-tab-item:: Mode 2

      * Set ``C symmetry order`` to 3
      * Set ``Refinement mode`` to *mode 2 - alignment of averages*
      * Set ``Class selection`` to a comma separated list of good classes, listing the reference class first
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
      
    .. md-tab-item:: Mode 3

      * Set ``Refinement mode`` to *mode 3 - alignment to reference*
      * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

If everything went well, you should have a good low-resolution model of the SARS-CoV-2 spike.

Session 2: Constrained single-particle tomography (CSPT)
========================================================

In this session, we will use 2D projections extracted from the **HIV-1 Gag dataset** to reconstruct and refine its structures to high-resolution.

Because this step is time-consuming and cannot be completed within a single session, we will use pre-calculated results to accelerate the workflow. To ensure consistency, we will begin by importing a dataset of 19,972 particles along with their corresponding alignments to generate an initial low-resolution reconstruction. We will then proceed with further refinement steps, including particle filtering, region-based refinement, and tilt-geometry refinement, to improve map resolution.

.. nextpyp:: Step 1: Import particles
  :collapsible: open
  
  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Set ``Detection method`` to *import*

  * Set ``Particle radius (A)`` to 50 

  * Set ``Import particle coordinates (*.spk)`` to */nfs/bartesaghilab/nextpyp/workshop/10164/particles*. Select :bdg-primary:`Choose Folder`

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

  Open the particle picking block and verify that the particles are in the correct location.

.. nextpyp:: Step 2: Import alignments
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Pickng` block) and select :bdg-primary:`Calculate reconstruction`

  * On the **Sample** tab 
    
    - Set ``Molecular weight (kDa)`` to 300 

    - Set ``Particle radius (A)`` to 150 

    - Set ``Symmetry`` to C6

  * On the **Extraction** tab

    - Set ``Box size (pixels/voxels)`` to 128 

    - Set ``Image binning`` to 2

  * On the **Alignments** tab

    - From the ``Import from`` dropdown menu, select *nextPYP (*.bz2)*

    Set ``Input parameter file (*.bz2)`` to */nfs/bartesaghilab/nextpyp/workshop/10164/particles/tomo-coarse-refinement-fg2v2MJLSY4Ui908_r01_02.bz2*

  * On the **Reconstruction** tab

    - Enable ``Apply dose weighting``

  * Go to the **Resources** tab

    - Set ``Split, Threads`` to 124

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/cspt.webp
      
      Constrained single-particle tomography (CSPT)

  * Click inside the :bdg-secondary:`Calculate reconstruction` block to see the results:
    
    * Notice that the **FSC near Nyquist is positive** indicating the presence of duplicate particles
    
    * The histogram of *Per-Particle Scores* shows a bimodal distribution that will allow to filter out bad particles

.. nextpyp:: Step 3: Particle filtering
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`Particle filtering`

  * On the **Particle filtering** tab

    - Set ``Input parameter file(*.bz2)`` to the ``*.bz2`` file that appears in the parent directory

    - Set ``Score threshold`` to 3.5

    - Set ``Min distance between particles (unbinned pixels)`` to 54

    - Enable ``Permanently remove particles``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step 4: Region-based refinement, tilt-geometry and particle orientation refinement
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle filtering` block) and select :bdg-primary:`3D refinement`

  * On the **Particle extraction** tab

    - Set ``Box size (pixels/voxels)`` to 256

    - Set ``Image binning`` to 1

  * On the **Particle scoring function** tab

    - Set ``Last tilt for refinement`` to 8 

    - Set ``Max resolution (A)`` to 4:3.5

    - From the ``Masking strategy`` dropdown menu, select ``from file``

    - Click the :fa:`search` icon to select the ``Shape mask (*.mrc)``, browse to */nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc*, and click :bdg-primary:`Choose File`

  * On the **Refinement** tab

    - Set ``Input parameter file (*.bz2)`` to the file *_r01_02_clean.bz2* file in the parent block

    - Set ``Last iteration`` to 3

    - Check ``Refine tilt-geometry``

    - Check ``Refine particle alignments`` 

    - Set ``Number of regions`` to 8,8,2 

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/regionbased.webp
      
      Region-based refinement splits tomograms into patches and refines them independently

  * Notice that there is no longer positive FSC correlation at Nyquist, the score distribution is unimodal, and map resolution has improved

.. nextpyp:: Step 5: Movie frame refinement (optional)
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`Movie refinement`

  * Go to the **Particle scoring function** tab

    - Set ``Last tilt for refinement`` to 4 

    - Set ``Max resolution (A)`` to 3.5

  * Go to the **Frame refinement** tab

    - Next to ``Input parameter file (*.bz2)`` click the :fa:`search` icon. Select the *_r01_03.bz2* file and click :bdg-primary:`Choose File`

    - Set ``Spatial sigma`` to 400

    - Set ``Time sigma`` to 16

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/movie_refinement.webp
      
      Refinement of individual tilt-frames


Day 3 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open
  
  * How to perform sub-volume averaging for ab-initio structure determination

  * How to import particle coordinates and alignments into ``nextPYP``

  * How to improve map resolution using the CSPT refinement framework

  ``nextPYP`` supports many other refinement-related tasks including building shape masks, per-particle CTF refinement, 3D classification, etc. For more details, see the :doc:`Tomography<../tutorials/tomo_empiar_10164>` and the :doc:`Classification<../tutorials/tomo_empiar_10304>` tutorials.

  :doc:`On day 4<dhvi_day4>` we will demonstrate ``nextPYP``'s functionality for post-processing and on-the-fly data processing.