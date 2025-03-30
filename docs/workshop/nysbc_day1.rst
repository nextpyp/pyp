#######################################
NYSBC course: nextPYP practical (day 1)
#######################################

This session shows how to use ``nextPYP`` to convert raw tilt-series from `EMPIAR-10164` into a ~4Å resolution structure of immature HIV-1 Gag protein. We will also cover pre-processing, tomogram reconstruction, and particle-picking for two other datasets representative of datatypes often processed in tomography. 

Datasets
========

  #. Immature Gag protein from HIV-1 Virus-Like Particles (`EMPIAR-10164 <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_)
  #. Ribosomes from whole *Mycoplasma pneumoniae* cells (`EMPIAR-10499 <https://www.ebi.ac.uk/empiar/EMPIAR-10499/>`_)
  #. Ribosomes from FIB-SEM milled mouse eplithelial cells (`EMPIAR-10987 <https://www.ebi.ac.uk/empiar/EMPIAR-10987/>`_)

Session 1: Pre-processing and particle picking
==============================================

In this session we will import frames, perform pre-processing and tomogram reconstruction, and pick particles for HIV VLPs together. We will also import workflows to pick ribosomes from whole *Mycoplasma* cells and lamellae cut from mouse epithelial cells. 

Create a new project
--------------------

.. nextpyp:: Data processing runs are organized into projects. We will create a new project for this tutorial
  :collapsible: open
  
  * Click on :bdg-primary:`Create new project`, give the project a name, and select :bdg-primary:`Create`

  * Select the new project from the **Dashboard** and click :bdg-primary:`Open`

  * The newly created project will be empty and a **Jobs** panel will appear on the right

Dataset 1: Immature Gag protein from HIV-1 VLPs
-----------------------------------------------

.. nextpyp:: Step 1: Import raw tilt-series
  :collapsible: open

  * Go to :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`  

  * A form to enter parameters will appear:

  * Go to the **Raw data** tab:

    - Set the ``path to raw data`` by clicking on the :fa:`search` icon and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/``
    
    - Type ``*.tif`` into the filter box (lower right) and click the :fa:`filter` icon
       
  * Go the the **Microscope Parameters** tab: 

    - Set ``Pixel size (A)`` to 1.35

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 85.3    

  * Click :bdg-primary:`Save` and the new block will appear on the project page. The block is in the modified state (indicated by the :fa:`asterisk` sign) and is ready to be executed

  * Clicking the :bdg-primary:`Run` button will show another dialog where you can select which blocks to run:

  * Click :bdg-primary:`Start Run for 1 block`. This will launch a process that reads one tilt at random and displays the resulting image inside the block

  * Click on the thumbnail inside the block to see a larger version of the projection image

.. nextpyp:: Step 2: Pre-processing
  :collapsible: open
  
  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

  * Go to the **Frame alignment** tab:

    - ``nextPYP`` uses the ``Frame pattern`` to extract metadata form the file names. EMPIAR-10164 follows the default file naming scheme and ``.tif`` extension, so we will leave the default setting. 

    - We will use ``unblur`` for frame alignment. 

  * Go to the **CTF determination** tab

    - Set ``Max resolution`` to 5 

  * Go to the **Tilt-series alignment** tab

    - Our ``Alignment method`` will be IMOD fiducial-based which is the default so make no changes.
  
  * Go to the **Tomogram reconstruction** tab
  
    - Our ``Reconstruction method`` will be IMOD, this is the default so make no changes. 

  * Go to the **Resources** tab

    - Set ``Threads per task`` to 41

    - Set ``Memory per task`` to 164
  
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel

  * When the block finishes running, examine the **Tilt-series**, **Plots**, **Table**, and **Gallery** tabs. We will measure our virions in this block as well.  

.. nextpyp:: Step 3: Particle picking
  :collapsible: open
    
  * We will be utilizing three separate blocks to perform geometrically constrained particle picking. This will allow for increased accruacy in particle detection and provides geometric priors for downstream refinement. 
  
  * Block 1: Virion selection
  
    * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

    * Go to the **Particle detection** tab:
      
      - Set ``Detection method`` to virions

      - Set ``Virion radius (A)`` to 500 (half the diameter we measured)
      
    * Click :bdg-primary:`Save`

  * Block 2: Virion segmentation

    * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

    * Click :bdg-primary:`Save`

  * Block 3: Spike (Gag) detection
  
    * Click on ``Segmentation (closed)`` (output of the :bdg-secondary:`Segmentation (closed surfaces)` block) and select :bdg-primary:`Particle-Picking (closed surfaces)`
    
    * Go to the **Particle detection** tab:
      
      - Set ``Detection method`` to uniform

      - Set ``Particle radius (A)`` to 50

      - Set ``Size of equatorial band to restrict spike picking (A)`` to 800
      
    * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel

Dataset 2: Ribosomes (whole *Mycoplasma* cells)
-----------------------------------------------

.. nextpyp:: Step 1: Import workflow
  :collapsible: open
  
  * In the upper left of your project page, click :bdg-primary:`Import Workflow`

  * Choose the **2025 NYSBC course: Pre-processing (EMPIAR-10499)** workflow by clicking the :bdg-primary:`Import` button to its right

  * We pre-set the parameters for the workflow, so you can immediately click :bdg-primary:`Save`. Three blocks will populate on the project page. 

.. nextpyp:: Step 2: Edit particle picking parameters
  :collapsible: open
  
  * Click into the settings of the :bdg-secondary:`Particle-Picking` block

    - Set ``Particle radius (A)`` to 80

    - Change ``Detection method`` from none to auto using the dropdown menu
  
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel

.. nextpyp:: Step 3: Copy particles and manually edit
  :collapsible: open
  
  * Click on the menu for the :bdg-secondary:`Particle-Picking` block

  * Select **Copy** 

  * Check **Copy files and data** and **Make automatically-picked particles editable** 

  * Click :bdg-primary:`Next`

  * Click into the new :bdg-secondary:`Particle-Picking` block. 

  * Ensure you are on the **Particles** tab. Here, you can right click to remove particles and left click to add particles. 

  * This manual picking feature is what I used the generate a particle set for nn-training for the next particle picking method we will use on the third dataset. 

Dataset 3: Ribosomes (lamellae from mouse epithelial cells)
-----------------------------------------------------------

.. nextpyp:: Step 1: Import workflow
  :collapsible: open
  
  * In the upper left of your project page, click :bdg-primary:`Import Workflow`

  * Choose the **2025 NYSBC course: Pre-processing (EMPIAR-10987)** workflow by clicking :bdg-primary:`Import`

  * We pre-set the parameters for the workflow, so you can immediately click :bdg-primary:`Save`. Three blocks will populate on the project page. 

.. nextpyp:: Step 2: Edit particle picking parameters
  :collapsible: open
  
  * Click into the settings of the :bdg-secondary:`Particle-Picking (eval)` block

    - Click the :fa:`search` icon. Browse to ``/nfs/bartesaghilab/nextpyp/workshop/10987/model_last_contrastive.pth``

    - Set ``Particle radius (A)`` to 100

    - Set ``Threshold for soft/hard positives`` to 0.5

    - Set ``Max number of particles`` to 700
  
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel

Session 2: 3D reconstruction and refinement
===========================================

In this session we will import 19,972 HIV-Gag protein particles, import initial reference-based alignments, then go through a condensed version of the 3D Refinement pipeline to attain an ~4Å resolution structure from 5,000 filtered particles. At a high level, we will be performing reference-based refinement, filtering particles, performing region-based refinement and tilt-geometry refinement, refining movie frames, and completing post-processing. Then we will demonstrate using ChimeraX to visualize our results. 

.. nextpyp:: Step 1: Import particles
  :collapsible: open
  
  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Set ``Detection method`` to import

  * Set ``Particle radius (A)`` to 50 

  * Click :fa:`search` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/particles``. Select :bdg-primary:`Choose Folder`

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step 2: Import alignments
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Pickng` block) and select :bdg-primary:`Particle refinement`

  * Go to the **Sample** tab 
    
    - Set ``Molecular weight (kDa)`` to 300 

    - Set ``Particle radius (A)`` to 150 

    - Set ``Symmetry`` to C6

  * Go to the **Extraction** tab

    - Set ``Box size (pixels/voxels)`` to 128 

    - Set ``Image binning`` to 2

  * Go to the **Refinement** tab

    - To demonstrate inserting a model, we will click the :fa:`search` icon next to ``Initial model (*.mrc)`` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_init_ref.mrc``  Click :bdg-primary:`Choose File`

    - Click the :fa:`search` icon next to ``Input parameter file (*.bz2)`` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/tomo-coarse-refinement-fg2v2MJLSY4Ui908_r01_02.bz2``  Click :bdg-primary:`Choose File`

    - Set the ``Max resolution (A)`` to 8

  * Go to the **Exposure weighting** tab

    - Turn ON ``Dose Weighting`` by checking the box 

  * Go to the **Resources** tab

    .. md-tab-set::

      .. md-tab-item:: I'm a core course participant

        - Set ``Threads per task`` to 124

        - Set ``Memory per task in GB`` to 720 

      .. md-tab-item:: I'm an additional TA

        - Set ``Threads per task`` to 70

        - Set ``Memory per task in GB`` to 720

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/cspt.webp
      :width: 75%
      :height: 75%
      
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
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle filtering` block) and select :bdg-primary:`Particle refinement`

  * Go to the **Extraction** tab

    - Set ``Box size (pixels/voxels)`` to 256

    - Set ``Image binning`` to 1

  * Go to the **Refinement** tab

    - Next to ``Initial model (*.mrc)`` click the :fa:`search` icon. Select the ``*_r01_01.mrc`` file and click :bdg-primary:`Choose File`

    - Next to ``Input parameter file (*.bz2)`` click the :fa:`search` icon. Select the ``_r01_02_clean.bz2`` file and click :bdg-primary:`Choose File`

    - Set ``Max resolution (A)`` to 4:3.5

    - Check ``Use signed correlation``

    - Set ``Last iteration`` to 3

    - Next to ``Shape mask (*.mrc)`` click the :fa:`search` icon. Browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc`` and click :bdg-primary:`Choose File`

  * Go to the **Constrained refinemnt** tab

    - Set ``Last exposure for refinement`` to 8 

    - Set ``Number of regions`` to 8,8,2 

    - Check ``Refine tilt-geometry``

    - Check ``Refine particle alignments`` 

  * Go to the **Exposure weighting** tab

    - Check ``Dose weighting`` (It may already be checked)

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/regionbased.webp
      :scale: 50%
      
      Region-based refinement

.. nextpyp:: Step 5: Movie frame refinement
  :collapsible: open
  
  * Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`Movie refinement`

  * Go to the **Refinement** tab

    - Next to ``Initial model (*.mrc)`` click the :fa:`search` icon. Select the ``*_r01_03.mrc`` file and click :bdg-primary:`Choose File`

    - Next to ``Input parameter file (*.bz2)`` click the :fa:`search` icon. Select the ``_r01_03.bz2`` file and click :bdg-primary:`Choose File`

    - Set ``Max resolution (A)`` to 3.5

  * Go to the **Constrained refinement** tab

    - Set ``Last exposure for refinement`` to 4 

    - Check ``Movie frame refinement`` 

    - Check ``Regularize translations`` 

    - Set ``Spatial sigma`` to 400

    - Set ``Time sigma`` to 16

    - If other boxes are checked, uncheck them 

  * Go to the **Exposure weighting** tab 

    - Check ``Dose weighting``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../images/workshop/movie_refinement.webp
      :width: 75%
      :height: 75%
      
      Refinement of individual tilt-frames

While the :bdg-secondary:`Movie refinement` block is running, we will demonstrate use of **ArtiaX** to visualize particle alignments

.. nextpyp:: 3D Visualization of alignments in ArtiaX
  :collapsible: open

  * For reference, these instructions are also available on the :doc:`User Guide<../guide/chimerax_artiax>`.
  
  * We assume the user already has the ArtiaX plugin, if not a simple google search will bring you to their docs for installation. 
  
  * Download files

    - Select a tomogram you wish to visualize the particles in. I will be using ``TS_43``. 
    
    - Click into the :bdg-secondary:`Pre-processing` block, go to **Tilt Series** tab and **Tomogram** sub tab. On this page, click the search icon, search for TS_43. Click the green button immediately above the tomogram display. This will download the tomogram in .rec format. 
    
    - Click into the :bdg-secondary:`Particle refinement` block, go to the **Metadata** tab. On this page, type ``TS_43`` into the search bar and click **Search**. Click the .star file to download particle alignments. 
    
    - Go to the **Reconstruction** tab and download the **Cropped Map**. 
    
  * Display in ChimeraX

    - Open ChimeraX (again, we assume ArtiaX is installed)
    
    - Open the tomogram ``TS_43.rec``
    
    - Run the following commands in the ChimeraX shell:
  
    .. code-block:: bash

      volume permuteAxes #1 xzy
      volume flip #2 axis z
        
    - Go to the **ArtiaX** tab and click **Launch** to start the plugin. 
    
    - In the **Tomograms** section on the left, select model #3 (permuted z flip) from the **Add Model** dropdown menu and click **Add!**
    
    - Go to the ArtiaX options panel on the right, and set the **Pixel Size** for the **Current Tomogram** to 10.8 (The current binned pixel size) 
    
    - On the left panel, under the **Particles List** section, select **Open List ...** and open the .star file. 
    
    - Return to the panel on the right and select the **Select/Manipulate** tab. Set the **Origin** to 1.35 (the unbinned pixel size)
    
    - From the **Color Settings** section, select **Colormap** and then **rlnLogLikelihoodContribution** from the dropdown menu. 
    
    - Play with the **Marker Radius** and **Axes Size** sliders to visualize the particle locations, cross correlation scores, and orientations.

.. nextpyp:: Step 6: Post-processing
  :collapsible: open
  
  * Click on ``Frames`` (output of the :bdg-secondary:`Movie refinement` block) and select :bdg-primary:`Post-processing`

  * Go to the **Post-processing** tab

    - Next to ``First half map (*_half1.mrc)`` click the :fa:`search` icon. Select the ``*_half1.mrc`` file and click :bdg-primary:`Choose File`

    - Set ``Masking method`` to from file usign the dropdown menu

    - Next to ``Mask file (*.mrc)`` click the :fa:`search` icon. Browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc`` and click :bdg-primary:`Choose File`

    - Set the ``B-factor method`` to adhoc using the dropdown menu

    - Set the ``Adhoc value (A^2)`` to -25 

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Map and model assessment in ChimeraX
  :collapsible: open
  
  * I will be using a prealigned pdb file and files downloaded from nextPYP to demonstrate how one can visualize their final map aligned to a model in Chimera. 

  * Download files

    - In the :bdg-secondary:`Post-processing` block, go to the **Reconstruction** tab. Click on the drop down menu **Select an MRC file to download**. Select the Full-Size Map. Your browser will download the post processed map as an MRC file. 

    - We are using a pre-aligned, pre-cropped pdb file (5L93) so do not need to download this. For your experiments, you would download whatever model required. 
  
    - Open the downloaded MRC file in Chimera. Visualize your beautiful map. To get a better look at your map/model fitting, open an atomic model in Chimera. Under the **Map** tab, Click **Zone**. Note we are left with a slightly larger zone than we would like so we will copy the zone command from the output to the terminal line, and edit the range. This leaves us with: 

    .. code-block:: bash 

      volume zone #2 nearAtoms #1 range 2.4

    - Select the model, go to **Actions**, **Atoms/Bonds**, and **Show Sidechain/Base**
    
    - You can now view the model fit to your map interactively in ChimeraX

Day 1 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  In this session we learned some of the things we are capable of doing in ``nextPYP``:
  
  * Raw data import

  * Pre-processing (frame alignment, tilt-series alignment, CTF estimation)

  * Tomogram reconstruction (WBP, fakeSIRT, SART)

    - ``nextPYP`` also supports :doc:`tomogram denoising<../guide/denoising>` using cryoCARE, IsoNet and Topaz Denoise

  * Segmentation (closed surfaces)

    - ``nextPYP`` also supports :doc:`open surface segmentation<../guide/segmentation>` which uses membrain-seg

  * Particle picking (geometrically constrained, size-based, nn-based, manual)

    - ``nextPYP`` also supports :doc:`template-search<../guide/picking3d>` and :doc:`molecular pattern mining<../guide/milopyp>`

  * Particle refinement (constrained single particle tomography, particle filtering, exposure weighting, region-based refinement, movie frame refinement, and post-processing)

    - ``nextPYP`` also supports particle-based CTF refinement, building shape masks, ab-initio refinement, and 3D classification

  We encourage you to explore the things we learned today as well as the other options available in ``nextPYP``. :doc:`Tomorrow<nysbc_day2>`, we will demonstrate ``nextPYP``'s functionality for on-the-fly data pre-processing and give you a chance to ask questions.