================================
Visualization in ChimeraX/ArtiaX
================================

``nextPYP`` produces all the necessary files to visualize a refined map into the original tomogram positions using ChimeraX/ArtiaX.

Step 1: Download the necessary files
------------------------------------

Select the refinement block where you will extract the particle locations from. Click on the ``Edit`` menu for that block and select the ``Show filesystem location`` to find out the location of the project in the filesystem.

From this location, you will need to download the following files to your local computer:

- Tomogram reconstruction (``mrc/tile_series_name.rec``)
- Refined map (``frealign/maps/block_name_r{class_number}_{iteration_number}_crop.mrc``)
- Corresponding particle orientations (``frealign/artiax/tile_series_name_K1.star``)

Step 2: Load data into ChimeraX/ArtiaX
--------------------------------------

    - Open ChimeraX (we assume the ArtiaX plugin is already installed)
    - Open the tomogram file ``tilt_series_name.rec``
    - Run the following commands in the ChimeraX shell:
       - ``volume permuteAxes #1 xzy``
       - ``volume flip #2 axis z`` (this step is only necessary when the dataset has virions)
    - Go to the ArtiaX tab and ``Launch`` the plugin
    - In the **Tomograms** section (main ArtiaX panel on the left), select model #3 (permuted z flip) from the ``Add Model`` dropdown menu and click ``Add!``
    - Go to to the ArtiaX options panel on the right, and set the ``Pixel Size`` for the **Current Tomogram** to the binned pixel size (10.8 for the EMPIAR-10164 tutorial) and click ``Apply``
    - From the **Particles List** section (main ArtiaX panel on the left), select ``Open List ...`` and browse to the location of the ``.star`` file
    - Go to the ArtiaX options panel on the right, select the **Select/Manipulate** tab and set the ``Origin`` of the ``Pixelsize Factors`` to the unbinned pixel size of the data (1.35 for the EMPIAR-10164 tutorial)
    - From the **Geometric Models** section (main ArtiaX panel on the left), go to ``Open Geomodel ...`` and select the refined map
    - Go to the ArtiaX options panel on the right, in the **Visualization** tab, set the ``Use Model`` from the dropdown menu to the refined map and click ``Attach Model``

If everything went well, you should obtain a result similar to this:

.. figure:: ../images/guide_artiax_10164.webp
    :alt: ArtiaX visualization of HIV1-Gag

.. tip::

    Depending on the dimensions of the refined map and the number of particles in the tomogram, you may need to downsample the map to make the program more responsive

.. seealso::

    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Particle picking<picking>`
    * :doc:`Neural-network picking<neural_network>`
    * :doc:`Overview<overview>`