##########################################
NYSBC Workshop nextPYP Practical (Day One)
##########################################

This session shows how to use :fa:`nextPYP` to convert raw tilt-series from `EMPIAR-10164` into a ~4Å resolution structure of immature HIV-1 Gag protein. We will also cover pre-processing, tomogram reconstruction, and particle-picking for two other datasets representative of datatypes often processed in tomography. 

Datasets
-------

  * EMPIAR-10164: HIV Virus-Like Particles (purified VLPs)
  * EMPIAR-10499: *Mycoplasma pneumoniae* cells (whole cells on grids) 
  * EMPIAR-10987: Mouse eplithelial cells (FIB-SEM milled lamellae)

Session Goal: Pre-Processing and Particle Picking
-------------------------------------------------
In this session we will import frames, perform pre-processing and tomogram reconstruction, and pick particles for on HIV VLPs together. We will also import workflows to pick ribosomes from bacteria cells and lamellae cut from mouse epithelial cells. 


Create a new project
--------------------

.. nextpyp:: Data processing runs are organized into projects. We will create a new project for this tutorial
  :collapsible: open

  * Click on :bdg-primary:`Create new project`, give the project a name, and select :bdg-primary:`Create`

  * Select the new project from the **Dashboard** and click :bdg-primary:`Open`

  * The newly created project will be empty and a **Jobs** panel will appear on the right

Dataset 1: EMPIAR-10164: HIV VLPs (Gag Protein)
----------------------------------------------

.. nextpyp:: Step 1: Import raw tilt-series 

  * Go to :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`  

  * A form to enter parameters will appear:

  * Go to the **Raw data** tab:

    - Set the ``path to raw data`` by clicking on the icon :fa:`search` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/``
    
    - Type ``*.tif`` into the filter box (lower right) and click the icon :fa:`filter`
       
  * Go the the **Microscope Parameters** tab: 

    - Set ``Pixel size (A)`` to 1.35

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 85.3    

  * Click :bdg-primary:`Save` and the new block will appear on the project page

  * The block is in the modified state (indicated by the :fa:`asterisk` sign) and is ready to be executed

  * Clicking the button :bdg-primary:`Run` will show another dialog where you can select which blocks to run:

  * Click :bdg-primary:`Start Run for 1 block`. This will launch a process that reads one tilt at random and displays the resulting image inside the block

  * Click on the thumbnail inside hte block to see a larger version of hte projection image

.. nextpyp:: Step 2: Pre-Processing and Tomogram Reconstruction

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

.. nextpyp:: Step 3: Particle Picking
  
  * We will be utilizing three separate blocks to perform geometrically constrained particle picking. This will allow for increased accruacy in particle detection and provides geometric priors for downstream refinement. 
  
  * Block One: Virion Selection
  
    * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

    * Go to the **Particle detection** tab:
      
      - Set ``Detection method`` to virions

      - Set ``Virion radius (A)`` to 500 (half the diameter we measured)
      
    * Click :bdg-primary:`Save`

  * Block Two: Virion Segmentation

    * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Picking` block) and select :bdg-primary:`Segmentation (closed surfaces)`

    * Click :bdg-primary:`Save`

  * Block Three: Spike (Gag) Detection
  
    * Click on ``Segmentation (closed)`` (output of the :bdg-secondary:`Segmentation (closed surfaces)` block) and select :bdg-primary:`Particle-Picking (closed surfaces)`
    
    * Go to the **Particle detection** tab:
      
      - Set ``Detection method`` to uniform

      - Set ``Particle radius (A)`` to 50

      - Set ``Size of equatorial band to restrict spike picking (A)`` to 800
      
    * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel
    


Dataset 2: EMPIAR-10499: Whole *Mycoplasma* Cells (Ribosomes)
------------------------------------------------------------

.. nextpyp:: Import Workflow

  * In the upper left of your project page, click :bdg-primary:`Import Workflow`

  * Choose the **2025 NYSBC workshop: Pre-processing (EMPIAR-10499)** workflow by clicking :bdg-primary:`Import`

  * We pre-set the parameters for the workflow, so you can immediately click :bdg-primary:`Save`. Three blocks will populate on the project page. 

.. nextpyp:: Edit Particle Picking Parameters

  * Click into the settings of the :bdg-primary:`Particle-Picking` block

    - Set ``Particle radius (A)`` to 80

    - Change ``Detection method`` from none to auto using the dropdown menu
  
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel

.. nextpyp:: Copy Particles and Manually Edit

  * Click on the menu for the :bdg-primary:`Particle-Picking` block

  * Select **Copy** 

  * Check **Copy files and data** and **Make automatically-picked particles editable** 

  * Click :bdg-primary:`Next`

  * Click into the new :bdg-primary:`Particle-Picking` block. 

  * Ensure you are on the **Particles** tab. Here, you can right click to remove particles and left click to add particles. 

  * This manual picking feature is what I used the generate a particle set for nn-training for the next particle picking method we will use on the third dataset. 

Dataset 3: EMPIAR-10987: FIB-SEM Milled Mouse Epithelial Cells (Ribosomes)
-------------------------------------------------------------------------

.. nextpyp:: Import Workflow

  * In the upper left of your project page, click :bdg-primary:`Import Workflow`

  * Choose the **2025 NYSBC workshop: Pre-processing (EMPIAR-10987)** workflow by clicking :bdg-primary:`Import`

  * We pre-set the parameters for the workflow, so you can immediately click :bdg-primary:`Save`. Three blocks will populate on the project page. 

.. nextpyp:: Edit Particle Picking Parameters

  * Click into the settings of the :bdg-primary:`Particle-Picking (eval)` block

    - Click the :fa:`search` icon. Browse to ``/nfs/bartesaghilab/nextpyp/workshop/10987/model_last_contrastive.pth``

    - Set ``Particle radius (A)`` to 100

    - Set ``Threshold for soft/hard positives`` to 0.5

    - Set ``Max number of particles`` to 700
  
  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 3 blocks`. Follow the status of the run in the **Jobs** panel

Session Goal: 3D Refinement
--------------------------

* In this session we will import 19,972 HIV-Gag protein particles, import initial reference-based alignments, then go through a condensed version of the 3D Refinement pipeline to attain an ~4Å resolution structure from 5,000 filtered particles. For the sake of time, we have pre-populated a workflow with parameters. As a group, we will import this work flow, then we will go through the steps and discuss the parameters and features while the refinement runs. 

.. nextpyp:: Step one: Import particles

  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Set ``Detection method`` to import

  * Set ``Particle radius (A)`` to 50 

  * Click :fa:`search` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/particles``. Select :bdg-primary:`Choose Folder`

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step two: Import alignments

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle-Pickng` block) and select :bdg-primary:`Particle refinement`

  * Go to the **Sample** tab 
    
    - Set ``Molecular weight (kDa)`` to 300 

    - Set ``Particle radius (A)`` to 150 

    - Set ``Symmetry`` to C6

  * Go to the **Extraction** tab

    - Set **Box size (pixels/voxels)** to 128 

    - Set **Image binning** to 2

  * Go to the **Refinement** tab

    - To demonstrate inserting a model, we will click the :fa:`search` icon next to ``Initial model (*.mrc)`` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_init_ref.mrc``  Click :bdg-primary:`Choose File`

    - Click the :fa:`search` icon next to ``Input parameter file (*.bz2)`` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/tomo-coarse-refinement-fg2v2MJLSY4Ui908_r01_02.bz2``  Click :bdg-primary:`Choose File`

    - Set the ``Max resolution (A)`` to 8

  * Go to the **Exposure weighting** tab

    - Turn ON ``Dose Weighting`` by checking the box 

  * Go to the **Resources** tab

    - Set ``Threads per task`` to 124

    - Set ``Memory per task in GB`` to 720 

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step three: Particle Filtering

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`Particle filtering`

  * Go to the **Particle filtering** tab

    - Set ``Score threshold`` to 3.5

    - Set ``Min distance between particles (unbinned pixels)`` to 54

    - Click the :fa:`search` icon next to ``Input parameter file(*.bz2)`` and select the ``*.bz2`` file that appears (this is from the parent directory). Click :bdg-primary:`Choose File`

    - Check the box next to ``Permanently remove particles``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step four: Region-based refinement, Tilt-geometry refinement, Further Particle refinement

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle filtering` block) and select :bdg-primary:`Particle refinement`

  * Go to the **Extraction** tab

    - Set ``Box size (pixels/voxels)`` to 256

    - Set ``Image binning`` to 1

  * Go to the **Refinement** tab

    - Next to ``Initial model (*.mrc)`` click the :fa:`search` icon. Select the ``*_r01_01.mrc`` file and click :bdg-primary:`Choose File`

    - Next to ``Input parameter file (*.bz2)`` click the :fa:`search` icon. Select the ``_r01_02_clean.bz2`` file and click :bdg-primary:`Choose File`

    - Set ``Max resolution (A)`` to 4:3

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

.. nextpyp:: Step five: Movie frame refinement

  * Click on ``Particles`` (output of the :bdg-secondary:`Particle refinement` block) and select :bdg-primary:`Movie refinement`

  * Go to the **Refinement** tab

    - Next to ``Initial model (*.mrc)`` click the :fa:`search` icon. Select the ``*_r01_03.mrc`` file and click :bdg-primary:`Choose File`

    - Next to ``Input parameter file (*.bz2)`` click the :fa:`search` icon. Select the ``_r01_03.bz2`` file and click :bdg-primary:`Choose File`

    - Set ``Max resolution (A)`` to 3

    - Check ``Use signed correlation`` if it is not already checked

    - Next to ``Shape mask (*.mrc)`` click the :fa:`search` icon. Browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc`` and click :bdg-primary:`Choose File`

  * Go to the **Constrained refinement** tab

    - Set ``Last exposure for refinement`` to 4 

    - Check ``Movie fream refinement`` 

    - Check ``Regularize translations`` 

    - If other boxes are checked, uncheck them 

  * Go to the **Exposure weighting** tab 

    - Check ``Dose weighting``

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

.. nextpyp:: Step six: Post-processing

  * Click on ``Frames`` (output of the :bdg-secondary:`Movie refinement` block) and select :bdg-primary:`Post-processing`

  * Go to the **Post-processing** tab

    - Next to ``First half map (*_half1.mrc)`` click the :fa:`search` icon. Select the ``*_half1.mrc`` file and click :bdg-primary:`Choose File`

    - Set ``Masking method`` to from file usign the dropdown menu

    - Next to ``Mask file (*.mrc)`` click the :fa:`search` icon. Browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/EMPIAR-10164_shape_mask.mrc`` and click :bdg-primary:`Choose File`

    - Set the ``B-factor method`` to adhoc using the dropdown menu

    - Set the ``Adhoc value (A^2)`` to -25 

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`




.. nextpyp:: Map/Model Assessment in Chimera (just watch, you can follow if you have Chimera with necessary plugins)

  * I will be using a prealigned pdb file and files downloaded from nextPYP to demonstrate how one can visualize their final map aligned to a model in Chimera. 

  * Download files

    - In the **Post-Processing** block, go to the **Reconstruction** tab. Click on the drop down menu **Select an MRC file to download**. Select the Full-Size Map. Your browser will download the post processed map as an MRC file. 

    - We are using a pre-aligned, pre-cropped pdb file (5L93) so do not need to download this. For your experiments, you would download whatever model required. 
  
    - Open the downloaded MRC file in Chimera. Visualize your beautiful map. To get a better look at your map/model fitting, open an atomic model in Chimera. Under the **Map** tab, Click **Zone**. Note we are left with a slightly larger zone than we would like so we will copy the zone command from the output to the terminal line, and edit the range. This leaves us with: 

    .. code-block:: bash 

      volume zone #2 nearAtoms #1 range 2.4

    - Select the model, go to **Actions**, **Atoms/Bonds**, and **Show Sidechain/Base**
    
    - You can now view the model fit to your map interactively in ChimeraX



.. nextpyp:: 3D Visualization in ArtiaX (just watch, though you can follow if you have ArtiaX plugin)

  * For future reference, these instructions are available on the nextPYP help page, under **User Guide**, and **3D Visualization (ArtiaX)**
  
  * We assume the user already has the ArtiaX plugin, if not a simple google search will bring you to their docs for installation. 
  
  * Download files

    - Select a tomogram you wish to visualize the particles in. I will be using TS_01. 
    
    - Click into the **Pre-processing** block, go to **Tilt Series** and **Tomogram**. On this page, click the search icon, search for TS_43. Click the green button immediately above the tomogram display. This will download the tomogram in .rec format. 
    
    - Click into the **Particle refinement** block, go to the **Metadata** tab. On this page, type **TS_43** into the search bar and click **Search**. Click the .star file to download particle alignments. 
    
    - Go to the **Reconstruction** tab of the **Particle refinement** block  and download the **Cropped Map**. 
    
  * Display in ChimeraX

    - Open ChimeraX (again, we assume ArtiaX is installed)
    
    - Open the tomogram **TS_01.rec** 
    
    - Run the following commands in the ChimeraX shell:
  
    .. code-block:: bash

      volume permuteAxes #1 xzy
      volume flip #2 axis z<h6>
        
    - Go to the **ArtiaX** tab and click **Launch** to start the plugin. 
    
    - In the **Tomograms** section on the left, select model #3 (permuted z flip) from the **Add Model** dropdown menu and click **Add!**
    
    - Go to the ArtiaX options panel on the right, and set the **Pixel Size** for the **Current Tomogram** to 10.8 (The current binned pixel size) 
    
    - On the left panel, under the **Particles List** section, select **Open List ...** and oepn the .star file. 
    
    - Return to the panel on the right and select the **Select/Manipulate** tab. Set the **Origin** to 1.35 (the unbinned pixel size)
    
    - From the **Color Settings** section, select **Colormap** and then **rlnLogLikelihoodContribution** from the dropdown menu. 
    
    - Play with the **Marker Radius** and **Axes Size** sliders to visualize the particle locations, cross correlation scores, and orientations. 




#########################################
NYSBC Workshop nextPYP Practical (Day Two)
#########################################

We will demonstrate how explicitly optimizing for fast runtime and giving users flexibility in pre-processing steps can aid in achieving high-quality and high-throughput data acquisition in nextPYP. Starting from **raw data** obtained at the microscope, we'll develop an **automatic pipeline** that can perform all **pre-processing** tasks up to and including particle picking. We will demonstrate this workflow on the EMPIAR-10164 dataset of HIV purified VLPs.

## Data

>EMPIAR-10164: HIV Virus-Like Particles (purified VLPs)


Create a Session
 
- On your Dashboard, select the blue **Go to Sessions** button.
- Click the blue **Start Tomography** button.



Session settings
 
- Give your session a user-readable name by typing in the ``Name`` box.
- The ``Parent Folder`` box will be auto-populated with the storage location specified in your ``pyp_config.toml`` file.
  - For the workshop, this is the ``/nfs`` mount for ``bartesaghilab``.
- Pick a *unique* ``Folder Name`` for your session. There can only be one folder name per session, regardless of the user-readable name!
- Select the ``Workshop`` group



  Raw data

- Path to raw data: ``/nfs/bartesaghilab/nextpyp/workshop/10164/TS_*.tif``



Microscope parameters

- Pixel size: 1.35
- Acceleration voltage: 300
- Tilt-axis angle: 85.3



Session settings

- Number of tilts: 41
- Raw data transfer: ``link``
  - ``Link``: Create a symlink between the data on the microscope and your local computer. The data still *only* exists at the microscope.
  - ``Move``: Transfer the data from the microscope to your local computer, removing the data at the microscope. The data will now *only* exist on your local computer.
  - ``Copy``: Make a copy of the data in the microscope, and transfer the copy to your local computer. The data will now exist at both the microscope *and* your local computer.



CTF determination

- Max resolution: 5



Virion detection

- Virion radius: 500
- Virion detection method: ``auto``
- Spike detection method: ``uniform``
- Minimum distance between spikes: 8
- Size of equatorial band to restrict spike picking: 800



Particle detection

- Detection method: ``none``
  - Remember that we have just picked our "particles" (virions) in the previous tab!
- Detection radius: 50



  Resources
  The following settings apply for all datasets:

  - Threads per task: 41
    - This number should match the number of tilts in your tilt series.
    - In general, the more threads you use, the more tilts that can be processed at the same time, and the faster you see pre-processing results.
  - Memory per task: 164
    - As a rule of thumb, use 4x as much memory as you have threads.
  


## More Features

  Using the Restart Option
 
  - "Smart" method of rerunning only what is necessary after changing pre-processing parameters
  - Workflow: Change a parameter → ``Save`` settings changes → ``Restart`` Pre-processing daemon
  - 
    Example: Changing the minimum distance between spikes

      - Virion detection
        - Increase ``Minimum distance between spikes (voxels)`` to 20
        - Click ``Save``
      - Navigate to ``Operations`` tab
      - Click ``Restart`` on pre-processing daemon
      - Open ``Logs`` to check that the restart flag has been detected and new pre-processing jobs will be launched in response to this change
      - Check ``Tilt series`` tab to see that fewer particles have been picked
    



  Using the Clear Option

  - Start pre-processing procedure from scratch
  - Helpful if the changes you've made touch multiple parts of the pre-processing pipeline
    - Like re-calculating CTF or re-doing frame alignment



  Navigating the Sessions homepage

  - Sessions can be **copied** or **deleted**
    - **CAUTION**: Deleting a session whose mode of file transfer was ``Move`` will **delete the data**.
  - Click the arrow to find where the session's network file storage location 
  
