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

Import Workflow

- We will follow much the same steps for EMPIAR-10987 as we did for EMPIAR-10499 and use blocks that we have pre-populated with runtime parameters for you. 
- In the upper left of your project page, click **Import Workflow**
- From the menu that populates, select the **Import** button to the right of **2025 NYSBC workshop: Pre-processing (EMPIAR-10987)**
- Click **Save**
- Three blocks should populate on your project page, **Tomgoraphy (from Raw Data)**, **Pre-processing**, and **Particle-Picking (eval)**. 
- Click **Run**, if only those 3 blocks are selected you can click **Start Run for 3 blocks**. If more than those three blocks are selected, deselect the extra blocks by clicking the blue checkbox to the left of the block name. Then click **Start Run for 3 blocks**. 

Particle Picking go through parameters

Session Goal: 3D Refinement
---------------------------
In this session we will import 19,972 HIV-Gag protein particles, import initial reference-based alignments, then go through a condensed version of the 3D Refinement pipeline to attain an ~4Å resolution structure from 5,000 filtered particles. For the sake of time, we have pre-populated a workflow with parameters. As a group, we will import this work flow, then we will go through the steps and discuss the parameters and features while the refinement runs. 

Show slides before hand what general steps are going to be (high level)


Import Workflow

- Click on the **Import Workflow** button in the top left of your project page. 
- Next to **2025 NYSBC workshop: end-to-end processing (EMPIAR-10164)** click the **Import** button. 
- Click **Save**, a series of 8 blocks should populate. 
- In the upper right hand corner, click **Run**, then **Start Run for 8 blocks**. 



End-To-End Processing

- Raw Data Import
- Pre-processing and Tomogram Generation
- Particle Picking (Imported)
- Initial Reference-Based Refinement (Imported)

.. figure:: ../images/workshop/cspt.webp
  :alt: Create new project

- Click into the block and navigate to the **Reconstruction** tab. You can see a variety of refinement statistics.

- Projections and slices of the reconstruction, FSC, per-projection statistics, per-particle statistics, per-particle scores
- Click on the **Expsoure Weights** tab to visualize the mean score per tilt over the order of acquisition. The weights are based on these scores. 
- Click on the **3D View** tab. In one of the drop down menus, select **Class 1, Iter 2**. Our initial map will populate. Alter the **Level of detail** and/or **Contour value** to sharpen the map, note we can visualize rough "sausages" representing our helices. 
- Particle Filtering


- Click into the **Particle filtering** block. 
- Navigate to the **Per-particle Scores** tab. Here we can visualize the global score cut off and where it lies on the score distribution for each of our tilt-series. Note it is settled nicely between the two peaks of the distribution. 

- Region-Based and Tilt Geometry Refinement

- Now we are used a filtered particle set to perform further refinement steps on. 
- Rather than clicking into the block, select the **Particle refinement** block menu at hte top right corner of the block. 
- From the drop down menu, select **Read** or **Edit** depending on if the block is still running or if it has finished. 
- Note the settings we have changed:

- In the **Sample** tab, we have applied C6 symmetry. 
- In the **Extraction** tab, we have reduced our binning to 1. 
- In the **Refinement** tab, we have increased our Max resolution (A) to 4 for the first iteration, and 3 for the second iteration. 
- In the **Constrained refinement** tab, we have applied a 8 regions in x and y and 2 regions in z. We have also turned ON Refine tilt-geometry 
- When the block is done running, click into the block. 
- Navigate to the **Reconstruction** tab and note immediately that we see details in both the projections and slices of our reeconstruction. Between our two iterations we see improvement in our FSC plots. We can clearly visualize defocus groups in our projection-based plots. Finally, We can also see that after filtering and further refinement, we no longer have a bimodal distribution in our Per-Particle Scores plot as we have removed all of the bad particles. 
- If one wanted to save plots from different blocks for say showing a supervisor during meetings, we are set up with Plotly and you can simply hover over the plot, then click the camera icon to download your plot in svg format for high resolution images. 


- Movie Frame Refinement

  - This step will optimize particle positions across frames, allowing for not only tilt-based refinement, but frame-based refinement. This is useful because the sample does not remain perfectly still across frame images and we can correction for this motion. 
  - Click into the **Movie refinement** block. 
  - I'll highlight some unique features of this block type, so click on the **Particle View** tab. 
  - Here you can click on a tilt-series image to enlareg it, and hit plus to further enlarge it. One can visualize the starting points (red), end points after movie frame refinement (yellow) and trajectories (green) of each particle identified on the 0 degree tilt. 
  - Click on the **Exposure Weights** tab. Click one of the plots to enlarge it. 
  - Here on the left you can see the mean score of each frame, with the first frame of each tilt highlighted in green. On the right is dose weighting by frame. 

- Post Processing

  - Once again, before we enter the block, click into the block settings. 
  - At this stage we are applying a mask and performing B-factor sharpening on our map. 
  - Return to the project page and click into the **Post-processing** block. 
  - Click on the **Reconstruction** tab. 
  - The final projections and slices should appear crisp with the corrected FSC showing a final resolution around 3.7 Ångstroms. 
  - Click on the **3D View** tab. 
  - If you remember, when we looked at our initial reconstruction after reference-based refienment, our helices looked like sausages. Now we can see definitive backbone density with some sidechain density as well. 
  - We will look at this map in Chimera now to view it along-side our model. 





Map/Model Assessment in Chimera (just watch, you can follow if you have Chimera with necessary plugins)

- I will be using a prealigned pdb file and files downloaded from nextPYP to demonstrate how one can visualize their final map aligned to a model in Chimera. 
- 
  Download files

  - In the **Post-Processing** block, go to the **Reconstruction** tab. Click on the drop down menu **Select an MRC file to download**. Select the Full-Size Map. Your browser will download the post processed map as an MRC file. 

  - We are using a pre-aligned, pre-cropped pdb file (5L93) so do not need to download this. For your experiments, you would download whatever model required. 
  
- Open the downloaded MRC file in Chimera. Visualize your beautiful map. To get a better look at your map/model fitting, open an atomic model in Chimera. Under the **Map** tab, Click **Zone**. Note we are left with a slightly larger zone than we would like so we will copy the zone command from the output to the terminal line, and edit the range. This leaves us with: 

 .. code-block:: bash 

  volume zone #2 nearAtoms #1 range 2.4

- Select the model, go to **Actions**, **Atoms/Bonds**, and **Show Sidechain/Base**
- You can now view the model fit to your map interactively in ChimeraX



3D Visualization in ArtiaX (just watch, though you can follow if you have ArtiaX plugin)

  - For future reference, these instructions are available on the nextPYP help page, under **User Guide**, and **3D Visualization (ArtiaX)**
  - We assume the user already has the ArtiaX plugin, if not a simple google search will bring you to their docs for installation. 
  - 
      Download files

      - Select a tomogram you wish to visualize the particles in. I will be using TS_01. 
      - Click into the **Pre-processing** block, go to **Tilt Series** and **Tomogram**. On this page, click the search icon, search for TS_43. Click the green button immediately above the tomogram display. This will download the tomogram in .rec format. 
      - Click into the **Particle refinement** block, go to the **Metadata** tab. On this page, type **TS_43** into the search bar and click **Search**. Click the .star file to download particle alignments. 
      - TODO: change to actual download, you can download a map in .mrc format from the **Reconstruction** tab of the **Particle refinement** block to attach to the particle locations. I will not be doing this today. 
    
  - 
      Display in ChimeraX

      - Open ChimeraX (again, we assume ArtiaX is installed)
      - Open the tomogram **TS_01.rec** 
      - Run the following commands in the ChimeraX shell:
  
   
        >volume permuteAxes #1 xzy
        >volume flip #2 axis z<h6>
        
      - Go to the **ArtiaX** tab and click **Launch** to start the plugin. 
      - In the **Tomograms** section on the left, select model #3 (permuted z flip) from the **Add Model** dropdown menu and click **Add!**
      - Go to the ArtiaX options panel on the right, and set the **Pixel Size** for the **Current Tomogram** to 10.8 (The current binned pixel size) 
      - On the left panel, under the **Particles List** section, select **Open List ...** and oepn the .star file. 
      - Return to the panel on the right and select the **Select/Manipulate** tab. Set the **Origin** to 1.35 (the unbinned pixel size)
      - From the **Color Settings** section, select **Colormap** and then **rlnLogLikelihoodContribution** from the dropdown menu. 
      - Play with the **Marker Radius** and **Axes Size** sliders to visualize the particle locations, cross correlation scores, and orientations. 





# NYSBC Workshop nextPYP Practical (Day Two)

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
  
