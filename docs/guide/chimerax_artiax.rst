===============================
Map particles back to tomograms
===============================

``nextPYP`` produces all the necessary files to visualize a refined map back into the original tomograms using the `ArtiaX <https://github.com/FrangakisLab/ArtiaX>`_ plugin for `ChimeraX <https://www.cgl.ucsf.edu/chimerax/>`_.

Download files
--------------

First, you will need to download the following files to your local computer:

- Tomogram reconstruction (``*.rec``)
- Refined map (``*.mrc``)
- Corresponding particle orientations (``*.star``)

You can get these files from ``nextPYP`` as follows:

- Go to the :bdg-secondary:`Pre-processing`` block, click on the **Tilt-series** tab, select a tilt-series and go to the **Reconstruction** tab. Download the ``.rec`` file by clicking on the gray/green badge
- Go to the :bdg-secondary:`Particle refinement` block, select the **Reconstruction** tab and select the ``Cropped Map`` option from the dropdown menu
- In the same :bdg-secondary:`Particle refinement` block, go the **Metadata** tab, type the name of the tilt-series in the box and click :bdg-primary:`Search`. Download the corresponding ``.star`` file by clicking on the gray/green badge

Display in ChimeraX
-------------------

- Open ChimeraX (we assume the ArtiaX plugin is already installed)
- Open the tomogram file ``tilt_series_name.rec``
- Run the following commands in the ChimeraX shell:
   - ``volume permuteAxes #1 xzy``
   - ``volume flip #2 axis z`` (this step is not always neccessary)
- Go to the ArtiaX tab and ``Launch`` the plugin
- In the **Tomograms** section (main ArtiaX panel on the left), select model #3 (permuted z flip) from the ``Add Model`` dropdown menu and click ``Add!``
- Go to to the ArtiaX options panel on the right, and set the ``Pixel Size`` for the **Current Tomogram** to the binned pixel size (10.8 for the EMPIAR-10164 tutorial) and click ``Apply``
- From the **Particles List** section (main ArtiaX panel on the left), select ``Open List ...`` and browse to the location of the ``.star`` file
- Go to the ArtiaX options panel on the right, select the **Select/Manipulate** tab and set the ``Origin`` of the ``Pixelsize Factors`` to the unbinned pixel size of the data (1.35 for the EMPIAR-10164 tutorial)
- From the **Geometric Models** section (main ArtiaX panel on the left), go to ``Open Geomodel ...`` and select the refined map
- Go to the ArtiaX options panel on the right, in the **Visualization** tab, set the ``Use Model`` from the dropdown menu to the refined map and click ``Attach Model``
- From the **Color Settings** section, select ``Colormap`` and choose the attribute ``rlnLogLikelihoodContribution`` from the dropdown menu

If everything went well, you should obtain a result similar to this:

.. figure:: ../images/guide_artiax_10164.webp
    :alt: ArtiaX visualization of HIV1-Gag

    Tomogram from immature HIV-1 virions from EMPIAR-10164 showing with a high-resolution model of Gag mapped back into the tomogram.

.. tip::

    - Repeat this process for other tilt-series in the dataset
    - Depending on the dimensions of the refined map and the number of particles in the tomogram, you may need to downsample the map to make ChimeraX more responsive