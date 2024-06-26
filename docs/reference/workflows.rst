=========
Workflows
=========

``nextPYP`` supports the use of pre-defined sequences of blocks, called **Workflows**. 

The program comes with workflows for the :doc:`Single-particle tutorial<../tutorials/spa_empiar_10025>`, the :doc:`Tomography tutorial<../tutorials/tomo_empiar_10164>` and the :doc:`Classification tutorial<../tutorials/tomo_empiar_10304>`.

1. Import a workflow
--------------------

- Go the **Dashboard**, create a new or open an existing project, then click on :fa:`project-diagram, text-primary` :badge:`Import Workflow,badge-primary`

- Choose a workflow from the list and click :badge:`Import,badge-primary`

- A form will appear asking for any required parameters. This typically includes the location of the raw data (and associated files) and the :doc:`Computing resources<computing>` to use

- Click :badge:`Save,badge-primary`, :badge:`Run,badge-primary`, and :badge:`Start Run for N blocks,badge-primary` (where N is the number of blocks in the workflow)

You can inspect the results of the **Workflow** by navigating into each block.

2. Defining custom workflows
----------------------------

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

.. seealso::

    * :doc:`Single-particle tutorial<../tutorials/spa_empiar_10025>`
    * :doc:`Tomography tutorial<../tutorials/tomo_empiar_10164>`
    * :doc:`Classification tutorial<../tutorials/tomo_empiar_10304>`
