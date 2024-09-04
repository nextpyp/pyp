==================================
MiLoPYP - Molecular pattern mining
==================================

Step 0: Pre-requisites
----------------------

To analyze the results of MiLoPYP, you first need to install `Phoenix-Arize <https://docs.arize.com/phoenix>`_ in your local machine. For macOS, follow these steps:

  * Download and install miniconda following `these <https://conda.io/projects/conda/en/latest/user-guide/install/macos.html>`_ instructions

  * Activate your miniconda installation, create a new conda environment and install Phoenix:
  
.. code-block:: bash

    source ${INSTALLATION_PATH}/miniconda3/bin/activate
	conda create -n "phoenix" python=3.10 -y
	conda activate phoenix
	conda install -c conda-forge arize-phoenix==0.0.28 pandas -y

Step 1: Training
----------------

    * Click on :guilabel:`Tomograms` (output of the :badge:`Pre-processing,badge-secondary` block) and select :badge:`MiLoPYP (train),badge-primary`

    * Select the training parameters

    * (optional) You can train on a subset of the tomograms by creating a `Filter <filters>`_ in the :badge:`Pre-processing,badge-secondary` block and selecting its name from the `Filter tomograms` dropdown menu

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. Follow the status of the run in the **Jobs** panel

    * Navigate to the :badge:`MiLoPYP (training),badge-primary` block to inspect the results of training


Step 2: Evaluation
-------------------

    * Click on :guilabel:`MiLoPYP model` (output of the :badge:`MiLoPYP (train),badge-secondary` block) and select :badge:`MiLoPYP (eval),badge-primary`

    * Select the trained model from the block upstream

    * Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for 1 block,badge-primary`. Follow the status of the run in the **Jobs** panel

    * Navigate to the :badge:`MiLoPYP (eval),badge-primary` block to inspect the results

.. note::

    Evalaution is always done on the entire set of tomograms available from the pre-processing block.

Step 3: Target selection
------------------------

  * Navigate to the :badge:`MiLoPYP (eval),badge-primary` block and download the **.tbz** file 

  * Decompress the **.tbz** file and run Phoenix

    .. code-block:: bash

        cd $WORK_DIRECTORY
        tar xvfz milopyp_interactive.tbz
    
  * Initiate a new shell and run: 
  
    .. code-block:: bash
    
        python -m http.server 7000

  * Download and run the visualization script: 
  
.. code-block:: bash

    wget https://raw.githubusercontent.com/nextpyp/cet_pick/main/cet_pick/phoenix_visualization.py
    python phoenix_visualization.py --input interactive_info_parquet.gzip



.. seealso::

    * :doc:`Particle picking<picking>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`