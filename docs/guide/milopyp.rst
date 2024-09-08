==================================
MiLoPYP - Molecular pattern mining
==================================

The ``MiLoPYP`` workflow consists of two main steps:

#. Molecular pattern minning (blocks :badge:`MiLoPYP (train),badge-primary` and :badge:`MiLoPYP (eval),badge-primary`)
#. Particle localization refinement (blocks :badge:`Particle-Picking (train),badge-primary` and :badge:`Particle-Picking (eval),badge-primary`)

.. figure:: ../images/milo_workflow.webp
    :alt: MiLoPYP workflow

Step 0: Pre-requisites
----------------------

Visualization
^^^^^^^^^^^^^

To analyze the results of ``MiLoPYP``, you will first need to install `Phoenix-Arize <https://docs.arize.com/phoenix>`_ in your local machine (even if you are accessing ``nextPYP`` remotely). 

For macOS, for example, follow these steps:

#. Download and install miniconda following `these <https://conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_ instructions

#. Activate your miniconda installation, create a new conda environment and install Phoenix:
  
.. code-block:: bash

    source ${INSTALLATION_PATH}/miniconda3/bin/activate
    conda create -n "phoenix" python=3.8 -y
    conda activate phoenix
    conda install -c conda-forge arize-phoenix==0.0.28 pandas -y

Data pre-processing
^^^^^^^^^^^^^^^^^^^

Since ``MiLoPYP`` operates on reconstructured tomograms, you first need to import and pre-process your data in ``nextPYP`` using a data import and :badge:`Pre-processing,badge-primary` blocks.

Step 1: Pattern mining training
-------------------------------

To train the exploration model:

#. Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`MiLoPYP (train),badge-primary`

#. Change the training parameters as needed

#. (optional) If you want to train MiLoPYP on a subset of the tomograms, create a :doc:`Filter<filters>` in the :badge:`Pre-processing,badge-secondary` block and select its name from the **Filter tomograms** dropdown menu

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`MiLoPYP (train),badge-primary` block to inspect the training metrics


Step 2: Pattern mining evaluation
---------------------------------

We can now evaluate the model to visualize the results:

#. Click on :guilabel:`MiLoPYP model` (output of the :badge:`MiLoPYP (train),badge-secondary` block) and select :badge:`MiLoPYP (eval),badge-primary`

#. Select the trained model from the block upstream (*.pth), for example, model_last_contrastive.pth

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`MiLoPYP (eval),badge-primary` block to inspect the results

.. note::

    Unlike training, Evaluation is always done on the entire set of tomograms

Step 3: Interactive target selection
------------------------------------

Next, we use Phoenix to interactively select the target particles:

#. Navigate to the :badge:`MiLoPYP (eval),badge-primary` block and download the file ***_milo.tbz** to your local machine

#. Open a terminal in your local machine, decompress the ***_milo.tbz** file, and run Phoenix:

.. code-block:: bash

    cd $WORK_DIRECTORY
    tar xvfz *_milo.tbz
    conda activate phoenix
    curl https://raw.githubusercontent.com/nextpyp/cet_pick/main/cet_pick/phoenix_visualization.py -o phoenix_visualization.py
    python phoenix_visualization.py --input interactive_info_parquet.gzip

If everything went ewll, you should see an output like this:

.. code-block:: bash

        name           coord                                         embeddings  label                             image
    0  TS_43   [299, 57, 96]  [-0.006966044, 0.014659109, -0.020045772, 0.00...     29  http://localhost:7000/imgs/0.png
    1  TS_43  [421, 145, 87]  [-0.024671286, 0.0323345, -0.06243068, 0.02977...     53  http://localhost:7000/imgs/1.png
    2  TS_43  [57, 267, 124]  [-0.016118556, 0.021317916, -0.044905104, 0.01...     29  http://localhost:7000/imgs/2.png
    3  TS_43  [288, 61, 104]  [-0.015271036, 0.024842143, -0.028918939, 0.00...     29  http://localhost:7000/imgs/3.png
    4  TS_43   [278, 71, 98]  [-0.022570543, 0.034957167, -0.03830565, 0.016...     29  http://localhost:7000/imgs/4.png
    🌍 To view the Phoenix app in your browser, visit http://localhost:57534/
    📺 To view the Phoenix app in a notebook, run `px.active_session().view()`
    📖 For more information on how to use Phoenix, check out https://docs.arize.com/phoenix

#. On another shell (in the same directory), activate the miniconda environment and start the image server: 
  
.. code-block:: bash

    conda activate phoenix
    cd $WORK_DIRECTORY
    python -m http.server 7000

#. Open a browser and visit the url as displayed above, for example: http://localhost:57534/

#. Under **Embeddings**, click on ``image_embedding`` and you will be able to visualize the results. Clicking on individual points will show the assocaited image in the bottom panel. You can also select a cluster using the left side bar to visualize a set of points (the correspoinding image gallery will be shown at the bottom of the page)

#. Select the points or clusters of interest using the **Select** tool

#. Export your selection using the **Export** button and **Download** the results as a ``.parquet`` file

.. note::

    By default, Phoenix's web server runs on port 7000. If that port is not available on your computer, you can specify a custom one using ``phoenix_visualization.py``'s ``--port`` option, for example, ``phoenix_visualization.py --input interactive_info_parquet.gzip --port 8000``. In this case, you will need to specify the same port number when running the http.server, for example, ``python -m http.server 8000``.

Step 4: Upload .parquet file to ``nextPYP``
-------------------------------------------

* Go back to ``nextPYP`` and navigate to the :badge:`MiLoPYP (eval),badge-primary` block

* Click on the Upload button :fa:`upload, text-primary`, browse to the location of your ``.parquet`` file, and upload the file

Step 5: Train particle refinement model
---------------------------------------

#. Click on :guilabel:`MiLoPYP Particles` (output of the :badge:`MiLoPYP (eval),badge-secondary` block) and select :badge:`Particle-Picking (train),badge-primary`

#. Select ``Import results from MiLoPYP`` and specify the ``.parquet`` file you uploaded in the previous step, for example, ``particles.parquet``

#. Set parameters for training as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking (eval),badge-primary` block to inspect the training metrics

Step 6: Evaluate particle refinement model
------------------------------------------

#. Click on :guilabel:`Particles Model` (output of the :badge:`Particle-Picking (train),badge-secondary` block) and select :badge:`Particle-Picking (eval),badge-primary`

#. Select a ``Trained model (*.pth)`` using the file browser

#. Set parameters for evaluation as needed

#. Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`

#. Navigate to the :badge:`Particle-Picking (eval),badge-primary` block to inspect the particle picking results

Having obtained a set of particles, we can now proceed to 3D refinement by connecting the output of the :badge:`Particle-Picking (eval),badge-primary` block to a new :badge:`Particle refinement,badge-secondary` block

.. seealso::

    * :doc:`Particle picking<picking>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`