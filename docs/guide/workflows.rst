=========
Workflows
=========

``nextPYP`` supports the use of pre-defined sequences of blocks, called **Workflows**. 

Pre-loaded workflows are included to execute the :doc:`Single-particle<../tutorials/spa_empiar_10025>`, :doc:`Tomography<../tutorials/tomo_empiar_10164>` and :doc:`Classification<../tutorials/tomo_empiar_10304>` tutorials.

Import a workflow
-----------------

- Go the **Dashboard**, create a new or open an existing project, then click on :fa:`project-diagram` :bdg-primary:`Import Workflow`

- Choose a workflow from the list and click :bdg-primary:`Import`

- A form will appear asking for any required parameters. This typically includes the location of the raw data (and associated files) and the compute :doc:`Resources<../reference/computing>` to use

- Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for N blocks` (where N is the number of blocks in the workflow)

You can inspect the results of the **Workflow** by navigating into each block.

Defining workflows
------------------

**Workflows** are defined using  ``.toml`` files. Here is an example:

.. code-block:: toml

    name = "Test workflow"
    description = """\
        Here goes the workflow description.
        """

    # comments here are useful to share info with other workflow authors looking at this file
    [blocks.rawdata]
    blockId = "sp-rawdata"
    name = "Raw Data"

    [blocks.rawdata.args]
    data_path = { ask = true } 
    scope_pixel = 0.66
    scope_voltage = 300

    [blocks.preprocessing]
    blockId = "sp-preprocessing"
    name = "Pre-processing"
    parent = "rawdata"

    [blocks.preprocessing.args]
    detect_rad = 75
    detect_method = "all"
    slurm_tasks = 7
    slurm_memory = 14

.. tip::

    You can use the workflows included with ``nextPYP`` as a starting point to create your own workflows.

The location of the ``.toml`` files must be specified in ``nextPYP``'s configuration file, using the entry ``workflowDirs``:

.. code-block:: toml

    workflowDirs = ["/path/to/workflows"] 