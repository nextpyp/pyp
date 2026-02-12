#############################
DHVI nextPYP tutorial (day 1)
#############################

Data import and pre-processing of tilt-series
=============================================

In this session, we will import raw data from three datasets from the EMPIAR database and perform movie-frame alignment, tilt-series alignment, tilted CTF estimation, and tomogram reconstruction.

Dataset 1: Immature Gag protein from HIV-1 Virus-Like Particles (`EMPIAR-10164 <https://www.ebi.ac.uk/empiar/EMPIAR-10164/>`_)
---------------------------------------------------------------------------------------------------------------------------

.. nextpyp:: Step 1: Create a project
  :collapsible: open
  
  Data processing runs are organized into projects. We will create a new project for this tutorial:

  * Click on :bdg-primary:`Create new project`, give the project a name, and select :bdg-primary:`Create`

  * Select the new project from the **Dashboard** and click :bdg-primary:`Open`

  * The newly created project will be empty and a **Jobs** panel will appear on the right

.. note::

  The number of projects you can create is limited to 1 to control the computational resources used by each user. On an actual production instance of nextPYP, you can create as many projects as you want.


.. nextpyp:: Step 2: Import raw tilt-series
  :collapsible: open

  * Go to :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`  

  * A form to enter parameters will appear:

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

  * Click on the thumbnail inside the block to see a larger version of the projection image

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

  * When the block finishes running, examine the **Tilt-series**, **Plots**, **Table**, and **Gallery** tabs. We will measure our virions in this block as well.  


Dataset 2: Spike preotein from SARS-CoV-2 virions (`EMPIAR-10453 <https://www.ebi.ac.uk/empiar/EMPIAR-10453/>`_)
-------------------------------------------------------------------------------------------------------------

We will use a subset of 10 tilt-series (049, 050, 071, 121, 162, 244, 271, 288, 291, and 297). The data were acquired using a Titan Krios with a K2 detector in counting mode. The pixel size is 1.329 Å and the tilt range is from -60° to +60° with a 3° increment.

.. nextpyp:: Step 1: Import raw tilt-series
  :collapsible: open

  * Click :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`

  * On the **Raw data** tab:

    - Set the ``Location`` by clicking on the :fa:`search` icon and browsing to ``/nfs/bartesaghilab/nextpyp/workshop_dhvi/10164/``

    - Type ``*.mrc`` into the filter box (lower right) and click the :fa:`filter` icon

  * On the **Microscope parameters** tab:

    - Set ``Pixel size (A)`` to 1.329

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to 84.8

  * Click :bdg-primary:`Save` and :bdg-primary:`Run` the block.

.. nextpyp:: Step 2: Pre-processing
  :collapsible: open

  We will create two pre-processing blocks with slightly different parameters as the best parameters for segmentation quality and particle picking can be different.

  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

  * **First block (for segmentation):**

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

      - Set ``High-frequency filtering`` to *"hamming (as in tomo3d)"*

    * On the **Resources** tab

      - Set ``Split, Threads`` to 11

  * **Second block (for particle picking):**

    Use the same parameters as the previous block except:

    * On the **Tomogram reconstruction** tab:

      - Set ``Radial filtering`` to *"fakeSIRT (mimic SIRT reconstruction)"*

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel


Dataset 3: HA spike from Influenza virions (`EMPIAR-12864 <https://www.ebi.ac.uk/empiar/EMPIAR-12864/>`_)
---------------------------------------------------------------------------------------------------------

We will use a subset of 3 tilt-series (TS4_58, TS7_62, and TS18_73). The data were acquired using a Titan Krios with a K3 detector in super-resolution mode. The pixel size is 2.0873 Å and the tilt range is from -60° to +60° with a 3° increment.

.. nextpyp:: Step 1: Import raw tilt-series
  :collapsible: open

  * Click :bdg-primary:`Import Data` and select :bdg-primary:`Tomography (from Raw Data)`

  * On the **Raw data** tab:

    - Set the ``Location`` by clicking on the :fa:`search` icon and browsing to ``/nfs/bartesaghilab/nextpyp/workshop_dhvi/12864/``

    - Type ``*.tif`` into the filter box (lower right) and click the :fa:`filter` icon

  * On the **Microscope parameters** tab:

    - Set ``Pixel size (A)`` to 1.04375

    - Set ``Acceleration voltage (kV)`` to 300

    - Set ``Tilt-axis angle (degrees)`` to -85

  * Click :bdg-primary:`Save` and :bdg-primary:`Run` the block.

.. nextpyp:: Step 2: Pre-processing
  :collapsible: open

  * Click on ``Tilt-series`` (output of the :bdg-secondary:`Tomography (from Raw Data)` block) and select :bdg-primary:`Pre-processing`

  * On the **Frame alignment** tab:

    - Leave all defaults

    - Set ``Binning`` to 4

  * On the **Tilt-series alignment** tab:

    - Disable ``Auto binning``

    - Set ``Binning for tilt-series alignment`` to 2

    - Disable ``Reshape tilt-images into squares``

    - Set ``Fiducial diameter (A)`` to 75

  * On the  **CTF determination** tab:

    - Set ``Max resolution (A)`` to 10

    - Set ``Min defocus (A)`` to 45000

    - Set ``Max defocus (A)`` to 75000

  * On the **Tomogram reconstruction** tab:

    - Set ``Tomogram thickness (unbinned voxels)`` to 3072

    - Set ``Binning factor`` to 16

    - Set ``Erase fiducials``

    - Set ``Generate half-tomograms``

  * On the **Resources** tab

    - Set ``Split, Threads`` to 11

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`. Follow the status of the run in the **Jobs** panel


Day 1 summary
=============

.. nextpyp:: What we learned today
  :collapsible: open

  In this session we learned some of the things we are capable of doing in ``nextPYP``:
  
  * Raw data import

  * Pre-processing (frame alignment, tilt-series alignment, CTF estimation)

  * Tomogram reconstruction (WBP, fakeSIRT, SART)

  :doc:`On day 2<dhvi_day2>` we will demonstrate ``nextPYP``'s functionality for denosing and particle picking.