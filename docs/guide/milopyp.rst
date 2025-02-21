=======================
Pattern Mining (MiLoPYP)
=======================

`MiLoPYP <https://nextpyp.app/milopyp/>`_ can be used to map the contents of a set of tomograms, with the goal of identifying targets of interest for sub-tomogram averaging as described in `Huang et al. <https://cryoem.cs.duke.edu/node/milopyp-self-supervised-molecular-pattern-mining-and-particle-localization-in-situ/>`_.

The ``MiLoPYP`` workflow in ``nextPYP`` consists of two steps and is implemented using four blocks:

#. **Pattern minning** uses the :bdg-secondary:`MiLoPYP (train)` and :bdg-secondary:`MiLoPYP (eval)` blocks 
#. **Position refinement** uses the :bdg-secondary:`Particle-Picking (train)` and :bdg-secondary:`Particle-Picking (eval)` blocks

Here is an example of how the workflow looks in the project view (relevant blocks are highlighted in blue):

.. figure:: ../images/milopyp_workflow.webp
    :alt: MiLoPYP workflow

Pre-requisites
--------------

Visualization
^^^^^^^^^^^^^

To analyze the results of ``MiLoPYP`` interactively, you need to install and run `Phoenix-Arize <https://docs.arize.com/phoenix>`_ either `remotely <https://nextpyp.app/milopyp/explore/#3d-interactive-session>`_ or in your *local* machine.

For a local installation on macOS, for example, follow these steps:

#. Download and install miniconda following `these <https://conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_ instructions

#. Activate the miniconda installation, create a new conda environment and install Phoenix:
  
.. code-block:: bash

    source ${INSTALLATION_PATH}/miniconda3/bin/activate
    conda create -n "phoenix" python=3.8 -y
    conda activate phoenix
    conda install -c conda-forge arize-phoenix==0.0.28 pandas -y

Data pre-processing
^^^^^^^^^^^^^^^^^^^

Since ``MiLoPYP`` operates on reconstructed tomograms, you first need to pre-process your tilt-series using the :bdg-secondary:`Pre-processing` block (see examples in the :doc:`tomography<../tutorials/tomo_empiar_10164>` and :doc:`classification<../tutorials/tomo_empiar_10304>` tutorials)

Pattern mining (training)
-------------------------

To train the mining/exploration module:

#. Click on :guilabel:`Tomograms` (output of the :bdg-secondary:`Pre-processing` block) and select :bdg-primary:`MiLoPYP (train)`

#. Set the training parameters as needed

#. (optional) If you want to train MiLoPYP on a subset of the tomograms in your dataset, create a :doc:`Filter<filters>` in the :bdg-secondary:`Pre-processing` block and select its name from the **Filter tomograms** dropdown menu at the top of the form. For datasets with many tomograms, doing this will considerably speed up training

#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

#. Once the run completes, navigate to the :bdg-secondary:`MiLoPYP (train)` block to monitor the training metrics

Pattern mining (evaluation)
---------------------------

The trained model can now be evaluated to visualize the results:

#. Click on :guilabel:`MiLoPYP model` (output of the :bdg-secondary:`MiLoPYP (train)` block) and select :bdg-primary:`MiLoPYP (eval)`

#. Select the trained model from the block upstream (``*.pth``), for example, ``model_last_contrastive.pth``. The models will be saved in sub-folders named with the date and time of training: ``YYYYMMDD_HHMMSS``

#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

#. Once the run completes, navigate to the :bdg-secondary:`MiLoPYP (eval)` block to visualize the embedding and the cluster labels

.. figure:: ../images/milopyp_eval.webp
    :alt: MiLoPYP evaluation

Target selection
----------------

There are two ways to select target positions to train the refinement module:

Option A: Manual cluster selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option only requires specifying a list of cluster numbers as displayed in the **Class Labels** panel, and can be done within ``nextPYP`` without running any external tools

Option B: Interactive target selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option requires running the program `Phoenix-Arize <https://docs.arize.com/phoenix>`_ to interactively select locations of interest:

* Navigate to the :bdg-secondary:`MiLoPYP (eval)` block and download the file ``*_milo.tbz``

* Open a terminal in your local machine, decompress the ``*_milo.tbz`` file, and run Phoenix:

.. code-block:: bash

    cd $WORK_DIRECTORY
    tar xvfz *_milo.tbz
    conda activate phoenix
    curl https://raw.githubusercontent.com/nextpyp/cet_pick/main/cet_pick/phoenix_visualization.py -o phoenix_visualization.py
    python phoenix_visualization.py --input interactive_info_parquet.gzip

If everything went well, you should see an output like this:

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

On another shell (in the same directory), activate the miniconda environment and start the image server: 
  
.. code-block:: bash

    conda activate phoenix
    cd $WORK_DIRECTORY
    python -m http.server 7000

With Phoenix now running:

* Open a browser and visit the url as displayed above, for example: http://localhost:57534/

* Under **Embeddings**, click on ``image_embedding`` to visualize the results. Clicking on a point in the cloud will show the associated image in the bottom panel. You can also select a cluster of points using the left side bar (the corresponding image gallery will be shown at the bottom of the page)

* Select the points or clusters of interest using the **Select** tool

* Export your selection using the **Export** button and **Download** the results as a ``.parquet`` file

.. note::

    By default, Phoenix's web server runs on port 7000. If that port is not available on your computer, you can specify a custom one using ``phoenix_visualization.py``'s ``--port`` option, for example, ``phoenix_visualization.py --input interactive_info_parquet.gzip --port 8000``. In this case, you will need to specify the same port number when running the http.server, for example, ``python -m http.server 8000``.

* Go back to ``nextPYP`` and navigate to the :bdg-secondary:`MiLoPYP (eval)` block

* Click on the **Upload** button :fa:`upload`, browse to the location of the ``.parquet`` file you exported from Phoenix, and upload the file

.. note::

    Currently, the file will be uploaded and always be renamed to ``particles.parquet`` on the remote server. If a file with that name already exists, it will be overwritten with the new file

Particle refinement (training)
------------------------------

Now that we have identified our targets of interest, we will use them to train the refinement module:

* Click on :guilabel:`MiLoPYP Particles` (output of the :bdg-secondary:`MiLoPYP (eval)` block) and select :bdg-primary:`Particle-Picking (train)`

* **Option A**: From the ``Coordinates for training`` menu select "class labels from MiLoPYP" and specify a comma separated list of classes using the class IDs displayed in the **Class Labels** panel

* **Option B**: From the ``Coordinates for training`` menu select "parquet file from MiLoPYP", and specify the location of the ``.parquet`` file you uploaded in the previous step: ``particles.parquet``

* Set parameters for training as needed

* Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

* Once the run completes, navigate to the :bdg-primary:`Particle-Picking (eval)` block to inspect the training metrics

Particle refinement (evaluation)
--------------------------------

The last step is to evaluate the model and obtain the final particle positions on all tomograms in the dataset:

#. Click on :guilabel:`Particles Model` (output of the :bdg-secondary:`Particle-Picking (train)` block) and select :bdg-primary:`Particle-Picking (eval)`

#. Select the location of the ``Trained model (*.pth)`` using the file browser. The models will be saved in sub-folders named with the date and time of training: ``YYYYMMDD_HHMMSS``

#. Set parameters for evaluation as needed

#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`

#. Once the run completes, navigate to the :bdg-secondary:`Particle-Picking (eval)` block to inspect the particle picking results

The resulting set of particles can be used for 3D refinement using the :bdg-secondary:`Particle refinement` block (see examples in the :doc:`tomography<../tutorials/tomo_empiar_10164>` and :doc:`classification<../tutorials/tomo_empiar_10304>` tutorials)

.. tip::

    * To detect particles distributed along fibers or tubules, select ``Fiber mode``. This will group neighboring particles, fit a smooth trajectory to them, and re-sample positions along the fitted curve
