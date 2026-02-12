###################
DHVI course (day 3)
###################

Part 1: Sub-volume averaging and classification
===============================================

Traditional sub-tomogram alignment-through-classification is a powerful strategy for *de novo* structure determination. It involves iterative 3D classification, alignment, and averaging of sub-volumes as described in  `Bartesaghi et al., 2008 <https://doi.org/10.1016/j.jsb.2008.02.008>`_. Initially, homogeneous particle groups are identified through 3D classification and subsequently averaged in 3D. The resulting class averages are then aligned to one another and combined into high signal-to-noise (SNR) references, which can be used to align individual sub-volumes. The resulting 3D models can then serve as references for high-resolution refinements using 2D projections.

Modes of operation
------------------

The sub-tomogram averaging functionality in ``nextPYP`` is provided by the :bdg-secondary:`Sub-tomogram averaging` block, which supports four primary modes of operation:

#. **Mode 0 - Global averaging and iterative centering**. Computes a global average of all sub-volumes, which can then be used (optionally) as a reference to iteratively center all sub-volumes using translation-only alignment. To enhance accuracy and reduce model bias, a radially symmetrized global average can be used as the reference.

#. **Mode 1 - 3D classification**. Based on the most recent set of alignments from the previous mode, sub-volumes are clustered into discrete classes, and class averages are computed.

#. **Mode 2 - Class average alignment**. Class averages are aligned to each other using a user-specified reference class. The user also selects which classes to retain. After alignment, the selected classes are averaged to produce a new reference volume.

#. **Mode 3 - Sub-volume alignment to reference**. Individual sub-tomograms are aligned to the reference generated in the previous step. Rotational alignment can either be global (searching the entire SO(3) space) or restricted to in-plane rotations around a pre-determined normal direction. When possible, restricting the rotation space often results in more accurate alignments.

Masks and filters for alignment and classification
--------------------------------------------------

For all modes, you can configure masking and filtering settings:

- **Masking**: Specify a radius in the x, y, and z directions and the apodization width, all in binned pixels.

- **Filtering**: Set low-pass and high-pass filter cutoffs and decay parameters, expressed as fractions of the Nyquist frequency. For example, ``0.05,0.01`` sets the cutoff to ``0.05``and the decay to ``0.01`` (0 being the DC-component and 1 being Nyquist).

.. nextpyp:: Step 1: Sub-volume generation
  :collapsible: open

  #. To access sub-volumes for averaging, generate them via the :bdg-secondary:`Particle picking` block. In the **Particle extraction** tab, set the ``Sub-volume export format`` to `3davg`, and define the desired ``Sub-tomogram size (voxels)`` and ``Sub-tomogram binning``.
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

Part 2: Constrained single-particle tomography (CSPT)
=====================================================

Now, we will import 19,972 HIV-Gag protein particles, import initial reference-based alignments, then go through a condensed version of the 3D Refinement pipeline to attain an ~4Å resolution structure from 5,000 filtered particles. At a high level, we will be performing reference-based refinement, filtering particles, performing region-based refinement and tilt-geometry refinement. 

.. nextpyp:: Step 1: Import particles
  :collapsible: open
  
  * Click on ``Tomograms`` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`Particle-Picking`

  * Set ``Detection method`` to import

  * Set ``Particle radius (A)`` to 50 

  * Click :fa:`search` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/particles``. Select :bdg-primary:`Choose Folder`

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

    - Click the :fa:`search` icon next to ``Input parameter file (*.bz2)`` and browse to ``/nfs/bartesaghilab/nextpyp/workshop/10164/tomo-coarse-refinement-fg2v2MJLSY4Ui908_r01_02.bz2``  Click :bdg-primary:`Choose File`

  * Go to the **Reconstruction** tab

    - Select ``Apply dose weighting`` by checking the box 

  * Go to the **Resources** tab

    - Set ``Split, Threads`` to 124

  * Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

    .. figure:: ../../images/workshop/cspt.webp
      
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

    .. figure:: ../../images/workshop/regionbased.webp
      
      Region-based refinement


Part 3: Visualization of results in ArtiaX/ChimeraX
===================================================

.. nextpyp:: Step 1: Download all the necessary files
  :collapsible: open

  - Select a tomogram you wish to visualize the particles in. I will be using ``TS_43``. 
  
  - Click into the :bdg-secondary:`Pre-processing` block, go to **Tilt Series** tab and **Tomogram** sub tab. On this page, click the search icon, search for TS_43. Click the green button immediately above the tomogram display. This will download the tomogram in .rec format. 
  
  - Click into the :bdg-secondary:`Particle refinement` block, go to the **Metadata** tab. On this page, type ``TS_43`` into the search bar and click **Search**. Click the .star file to download particle alignments. 
  
  - Go to the **Reconstruction** tab and download the **Cropped Map**. 
    
.. nextpyp:: Step 2: Display in ChimeraX
  :collapsible: open

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