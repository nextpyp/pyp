##########################################
NYSBC Workshop nextPYP Practical (Day One)
##########################################

We will demonstrate the use of **nextPYP** to generate an ~4Å resolution structure of HIV-Gag protein from three tilt-series. In this practical we will import raw data, perform pre-processing tasks, reconstruct tomograms, pick particles, and perform refinement to generate a map. We will all complete this procedure together on **EMPIAR-10164** and import workflows for two other datasets that are representative of additional sample types commonly imaged using cryo-ET. 

Datasets
________

>EMPIAR-10164: HIV Virus-Like Particles (purified VLPs)<br>
>EMPIAR-10499: *Mycoplasma pneumoniae* cells (whole cells on grids) <br>
>EMPIAR-10987: Mouse eplithelial cells (FIB-SEM milled lamellae)

## Session Goal: Pre-Processing and Particle Picking

In this session we will import frames, perform pre-processing and tomogram reconstruction, and pick particles for on HIV VLPs together. We will also import workflows to pick ribosomes from bacteria cells and lamellae cut from mouse epithelial cells. 

Create a Project
----------------
 
- On your Dashboard, select the blue **Create new project** button.
- In the window that pops up, name your project and hit **Create**. 
- Select your project, and then click **Open**. 
</details>

### 1. EMPIAR-10164: HIV VLPs (Gag Protein)

<details open>
<summary>Import Raw Data</summary>

- Go to the upper left fo the project page, select **Import Data**
- Select **Tomography (from Raw Data)**
- Edit the following parameters:
  - <details open>
    <summary>Raw Data</summary>
  
    - Click the magnifying glass and set the path to raw data:
     /nfs/bartesaghilab/nextpyp/workshop/10164/
    - Type *.tif into the box on the bottom right and hit the filter icon.
    </details>   
  - <details open>
     <summary>Microscope Parameters</summary>
  
      - Pixel size (A): 1.35
      - Acceleration voltage (kV): 300
      - Tilt-axis angle (degrees): 85.3
    </details>
- Click **Save**
- In the upper right hand corner of the project page, select **Run**, then **Start Run for 1 block**.
</details>

<details open>
<summary>Pre-Processing and Tomogram Reconstruction</summary>
  
- At the bottom right of the **Tomography (from Raw Data)** block, select the blue button labeled **Tilt-series**. 
- From the drop down **Use Data** menu, select **Pre-processing**. 
- Examine the following tabs:
  - <details open>
    <summary>Frame Alignment</summary>

    - Our frames are **.tif** files.
    - We will be using **unblur** to perform frame alignment. 
    - These are the default settings, so we do not change anything on this tab. 
    </details>
  - <details open>
    <summary>CTF Determination</summary>
  
      - Change the **Max resolution (A)** to 5.
    </details>
  - <details open>
    <summary>Tilt-series alignment</summary>
  
      - We will use **IMOD fiducials** based alignment. 
      - This is the default so we do not change anything. 
    </details>
  - <details open>
    <summary>Tomogram reconstruction</summary>
  
    - We will use **IMOD** to reconstruct our tomograms. 
    - For this dataset, the default setting work so we will not change them.
    </details>
- Select **Save**
- In the upper right hand corner of your project page, select **Run**, then **Start Run for 1 block**. 
</details>

<details open>
  <summary>Particle Picking</summary>
  
  - We will be utilizing three steps in three separate blocks to perform geometrically constrained particle picking. This will allow for increased accruacy in particle detection and provides geomtric priors for downstream refinement. 
  - <details open>
    <summary>Step One: Virion Selection</summary>
  
    - On the bottom right of the **Pre-processing** block, select the blue button labeled **Tomograms**. 
    - From the drop down **Use Data** menu, select **Particle-Picking** 
    - <details open>
        <summary>Go to the Particle Detection tab and change the following parameters:</summary>
      
        - Detection method: virions
        - Virion radius (A): 500 
      </details>
    - Click **Save**, **Run**, and **Start Run for 1 block**. 

    </details>
  - <details open>
    <summary>Step Two: Virion Segmentation</summary>

    - Click the blue button on the Virion Selection block labeled **Particles** and fromt he drop down menu, select **Segmentation (closed surfaces)**
    - We will not change any parameters for this block, so you can click **Save**, **Run**, and **Start Run for 1 block**. 

    </details>
  - <details open>
    <summary>Step Three: Spike (Gag) Detection</summary>
  
    - Click the blue button on the Virion Segmentation block labeled **Segmentation (closed)** and from the drop down menu, select Particle-Picking (closed surfaces). 
    - <details open>
        <summary>Go to the Particle Detection tab and change the following parameters:</summary>
      
        - Detection method: uniform
        - Particle radius (A): 50
        - Minimum distance between spikes (voxels): 8
        - Size of equatorial band to restrict spike picking (A): 800
      </details>
    - Click **Save**, **Run**, and **Start Run for 1 block**
    </details>
</details>

### 2. EMPIAR-10499: Whole *Mycoplasma* Cells (Ribosomes)

<details open>
<summary>Import Workflow</summary>

- In the upper left of your project page, click **Import Workflow**
- From the menu that populates, select the **Import** button to the right of **2025 NYSBC workshop: Pre-processing (EMPIAR-10499)**
- We have pre-set the parameters for each block, so you can immediately click **Save**
- Three blocks should populate on your project page, **Tomgoraphy (from Raw Data)**, **Pre-processing**, and **Particle-Picking**. 
- Click **Run**, if only those 3 blocks are selected you can click **Start Run for 3 blocks**. If more than those three blocks are selected, deselect the extra blocks by clicking the blue checkbox to the left of the block name. Then click **Start Run for 3 blocks**. 
</details>

### 3, EMPIAR-10987: FIB-SEM Milled Mouse Epithelial Cells (Ribosomes)

<details open>
<summary>Import Workflow</summary>

- We will follow much the same steps for EMPIAR-10987 as we did for EMPIAR-10499 and use blocks that we have pre-populated with runtime parameters for you. 
- In the upper left of your project page, click **Import Workflow**
- From the menu that populates, select the **Import** button to the right of **2025 NYSBC workshop: Pre-processing (EMPIAR-10987)**
- Click **Save**
- Three blocks should populate on your project page, **Tomgoraphy (from Raw Data)**, **Pre-processing**, and **Particle-Picking (eval)**. 
- Click **Run**, if only those 3 blocks are selected you can click **Start Run for 3 blocks**. If more than those three blocks are selected, deselect the extra blocks by clicking the blue checkbox to the left of the block name. Then click **Start Run for 3 blocks**. 
</details>

## Session Goal: 3D Refinement

In this session we will import 19,972 HIV-Gag protein particles, import initial reference-based alignments, then go through a condensed version of the 3D Refinement pipeline to attain an ~4Å resolution structure from 6,388 filtered particles. 

<details open>
<summary>Particle Import</summary>


</details>

<details open>
<summary>Alignment Import</summary>


</details>

<details open>
<summary>Particle Filtering</summary>


</details>

<details open>
<summary>Tilt-geometry and Region-based Refinement</summary>


</details>

<details open>
<summary>Movie Frame Refinement</summary>


</details>

<details open>
<summary>Post-Processing</summary>


</details>

<details open>
<summary>Map/Model Assessment in Chimera (just watch, you can follow if you have Chimera with necessary plugins)</summary>

- I will be using a prealigned pdb file and files downloaded from nextPYP to demonstrate how one can visualize their final map aligned to a model in Chimera. 


</details>

<details open>
<summary>3D Visualization in ArtiaX (just watch, though you can follow if you have ArtiaX plugin)</summary>

  - For future reference, these instructions are available on the nextPYP help page, under **User Guide**, and **3D Visualization (ArtiaX)**
  - We assume the user already has the ArtiaX plugin, if not a simple google search will bring you to their docs for installation. 
  - <details open>
      <summary>Download files</summary>

      - Select a tomogram you wish to visualize the particles in. I will be using TS_01. 
      - Click into the **Pre-processing** block, go to **Tilt Series** and **Tomogram**. On this page, click the green button immediately above the tomogram display. This will download the tomogram in .rec format. 
      - Click into the **Particle refinement** block, go to the **Metadata** tab. On this page, type **TS_01** into the search bar and click **Search**. Click the .star file to download particle alignments. 
      - In your own experiments, you can download a map in .mrc format from the **Reconstruction** tab of the **Particle refinement** block to attach to the particle locations. I will not be doing this today. 
    </details>
  - <details open>
      <summary>Display in ChimeraX</summary>

      - Open ChimeraX (again, we assume ArtiaX is installed)
      - Open the tomogram **TS_01.rec** 
      - Run the following commands in the ChimeraX shell:
        ></br>
        >volume permuteAxes #1 xzy</br>
        >volume flip #2 axis z</br><h6>
        >*Model numbers assume the tomogram is the first and only thing you have open in ChimeraX, if this is not the case, adjust accordingly.* </br>
        ></br>
      - Go to the **ArtiaX** tab and click **Launch** to start the plugin. 
      - In the **Tomograms** section on the left, select model #3 (permuted z flip) from the **Add Model** dropdown menu and click **Add!**
      - Go to the ArtiaX options panel on the right, and set the **Pixel Size** for the **Current Tomogram** to 10.8 (The current binned pixel size) 
      - On the left panel, under the **Particles List** section, select **Open List ...** and oepn the .star file. 
      - Return to the panel on the right and select the **Select/Manipulate** tab. Set the **Origin** to 1.35 (the unbinned pixel size)
      - From the **Color Settings** section, select **Colormap** and then **rlnLogLikelihoodContribution** from the dropdown menu. 
      - Play with the **Marker Radius** and **Axes Size** sliders to visualize the particle locations, cross correlation scores, and orientations. 

      
</details>


# NYSBC Workshop nextPYP Practical (Day Two)

We will demonstrate how explicitly optimizing for fast runtime and giving users flexibility in pre-processing steps can aid in achieving high-quality and high-throughput data acquisition in nextPYP. Starting from **raw data** obtained at the microscope, we'll develop an **automatic pipeline** that can perform all **pre-processing** tasks up to and including particle picking. We will demonstrate this workflow on the EMPIAR-10164 dataset of HIV purified VLPs.

## Data

>EMPIAR-10164: HIV Virus-Like Particles (purified VLPs)<br>

<details open>
<summary>Create a Session</summary>
 
- On your Dashboard, select the blue **Go to Sessions** button.
- Click the blue **Start Tomography** button.
</details>

<details open>
<summary>Session settings</summary>
 
- Give your session a user-readable name by typing in the ```Name``` box.
- The ```Parent Folder``` box will be auto-populated with the storage location specified in your ```pyp_config.toml``` file.
  - For the workshop, this is the ```/nfs``` mount for ```bartesaghilab```.
- Pick a *unique* ```Folder Name``` for your session. There can only be one folder name per session, regardless of the user-readable name!
- Select the ```Workshop``` group
</details>

<details open>
  <summary>Raw data</summary>

- Path to raw data: ```/nfs/bartesaghilab/nextpyp/workshop/10164/TS_*.tif```
</details>

<details open>
<summary>Microscope parameters</summary>

- Pixel size: 1.35
- Acceleration voltage: 300
- Tilt-axis angle: 85.3
</details>

<details open>
<summary>Session settings</summary>

- Number of tilts: 41
- Raw data transfer: ```link```
  - ```Link```: Create a symlink between the data on the microscope and your local computer. The data still *only* exists at the microscope.
  - ```Move```: Transfer the data from the microscope to your local computer, removing the data at the microscope. The data will now *only* exist on your local computer.
  - ```Copy```: Make a copy of the data in the microscope, and transfer the copy to your local computer. The data will now exist at both the microscope *and* your local computer.
</details>

<details open>
<summary>CTF determination</summary>

- Max resolution: 5
</details>

<details open>
<summary>Virion detection</summary>

- Virion radius: 500
- Virion detection method: ```auto```
- Spike detection method: ```uniform```
- Minimum distance between spikes: 8
- Size of equatorial band to restrict spike picking: 800
</details>

<details open>
<summary>Particle detection</summary>

- Detection method: ```none```
  - Remember that we have just picked our "particles" (virions) in the previous tab!
- Detection radius: 50
</details>

<details open>
  <summary>Resources</summary>
  The following settings apply for all datasets:

  - Threads per task: 41
    - This number should match the number of tilts in your tilt series.
    - In general, the more threads you use, the more tilts that can be processed at the same time, and the faster you see pre-processing results.
  - Memory per task: 164
    - As a rule of thumb, use 4x as much memory as you have threads.
  
</details>

## More Features
<details open>
  <summary>Using the Restart Option</summary>
 
  - "Smart" method of rerunning only what is necessary after changing pre-processing parameters
  - Workflow: Change a parameter → ```Save``` settings changes → ```Restart``` Pre-processing daemon
  - <details>
    <summary>Example: Changing the minimum distance between spikes</summary>

      - Virion detection
        - Increase ```Minimum distance between spikes (voxels)``` to 20
        - Click ```Save```
      - Navigate to ```Operations``` tab
      - Click ```Restart``` on pre-processing daemon
      - Open ```Logs``` to check that the restart flag has been detected and new pre-processing jobs will be launched in response to this change
      - Check ```Tilt series``` tab to see that fewer particles have been picked
    </details>
</details>

<details open>
  <summary>Using the Clear Option</summary>

  - Start pre-processing procedure from scratch
  - Helpful if the changes you've made touch multiple parts of the pre-processing pipeline
    - Like re-calculating CTF or re-doing frame alignment
</details>

<details open>
  <summary>Navigating the Sessions homepage</summary>

  - Sessions can be **copied** or **deleted**
    - **CAUTION**: Deleting a session whose mode of file transfer was ```Move``` will **delete the data**.
  - Click the arrow to find where the session's network file storage location 
  
</details>