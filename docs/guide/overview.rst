==============
Overview of UI
==============

Dashboard and projects
======================

* The **Dashboard** :fa:`tachometer-alt` is the top-level page of ``nextPYP``

  .. figure:: ../images/dashboard_empty.webp
    :alt: Dashboard view

* **Projects** :fa:`project-diagram` are logical units within the dashboard and are used to organize data processing runs

  .. figure:: ../images/tutorial_tomo_open.webp
    :alt: Project view

Blocks
======

Each **Block** represents a unit of processing (data import, pre-processing, refinement, masking, etc.). They provide control over the corresponding processing parameters and an interface to visualize the results. Projects consist of sequences of connected blocks.

.. figure:: ../images/overview_blocks_view.webp
  :alt: Project view

.. tip::
  - Blocks can be moved around and organized within the canvas
  - Multiple blocks can be selected and moved as a set

Block connectivity
------------------

Each block in ``nextPYP`` has inputs and outputs of specific types. Two blocks can be connected only if their input/outputs types are compatible. The program does not allow for incompatible block connections.

Types
-----

Single-particle projects and tomography projects have similar block types. A list of block types and corresponding inputs and outputs is given below:

.. md-tab-set::

  .. md-tab-item:: Single-particle

    .. nextpyp:: :fa:`layer-group` Import

      **Description**: Import movie frames or micrographs in MRC, DM4, TIF, or EER format.
      
      **Input**: None
      
      **Output**: ``Movies``

    .. nextpyp:: :fa:`chart-bar` Pre-processing
  
      **Description**: Movie frame alignment, CTF estimation, and particle picking.
      
      **Input**: ``Movies``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`dot-circle` Particle refinement

      **Description**: Particle alignment, classification and per-particle CTF refinement.
      
      **Input**: ``Particles``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`filter` Particle filtering

      **Description**: Removal of bad particles from downstream analysis.
      
      **Input**: ``Particles``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`film` Movie refinement

      **Description**: Per-particle movie frame alignment and dose weighting.
      
      **Input**: ``Particles`` or ``Frames``
      
      **Output**: ``Frames``

    .. nextpyp:: :fa:`crop` Create mask

      **Description**: Creation of shape masks.
      
      **Input**: ``Particles`` or ``Frames``
      
      **Output**: ``None``

    .. nextpyp:: :fa:`star` Post-processing

      **Description**: Masking, map sharpening, and Fourier Shell Correlation (FSC) plots.
      
      **Input**: ``Particles`` or ``Frames``
      
      **Output**: ``None``

  .. md-tab-item:: Tomography

    .. nextpyp::  :fa:`cubes` Data import

      **Decription**: Import raw tilt-series data (with or without frames) in MRC, DM4, TIF, or EER format.
      
      **Input**: None
      
      **Output**: ``Tilt-series``

    .. nextpyp:: :fa:`chart-bar` Pre-processing (legacy)

      **Description**: Frame and tilt-series alignment, tomogram reconstruction, CTF estimation, and particle picking.
      
      **Input**: ``Tilt-series``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`chart-bar` Pre-processing

      **Description**: Frame and tilt-series alignment, tomogram reconstruction, and CTF estimation.
      
      **Input**: ``Tilt-series``
      
      **Output**: ``Tomograms``

    .. nextpyp:: :fa:`crosshairs` Denoising (train)

      **Description**: Train a neural network for tomogram denoising.
      
      **Input**: ``Tomograms``
      
      **Output**: ``Tomograms``

    .. nextpyp:: :fa:`crosshairs` Denoising (eval)

      **Description**: Evaluate a neural network for tomogram denoising.
      
      **Input**: ``Tomograms``
      
      **Output**: ``Tomograms``

    .. nextpyp:: :fa:`crosshairs` Segmentation (open surfaces)

      **Description**: 3D segmentation using pre-trained model.
      
      **Input**: ``Tomograms``
      
      **Output**: ``None``

    .. nextpyp:: :fa:`crosshairs` Particle-Picking

      **Description**: Import, manual, size-based, virions, or template-search particle picking.
      
      **Input**: ``Tomograms``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`search` MiLoPYP (train)

      **Description**: Train neural network for pattern mining.
      
      **Input**: ``Tomograms``
      
      **Output**: ``MiLoPYP Model``

    .. nextpyp:: :fa:`search` MiLoPYP (eval)

      **Description**: Evaluate neural network model for pattern mining.
      
      **Input**: ``MiLoPYP Model``
      
      **Output**: ``MiLoPYP Particles``

    .. nextpyp:: :fa:`crosshairs` Particle-Picking (train)

      **Description**: Train neural network for particle picking.
      
      **Input**: ``Particles``, ``MiLoPYP Particles``
      
      **Output**: ``Particles Model``

    .. nextpyp:: :fa:`crosshairs` Particle-Picking (eval)

      **Description**: Evaluate neural network for particle picking.
      
      **Input**: ``Tomograms``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`dot-circle` Particle refinement

      **Description**: Constrained particle alignment and classification, region-based refinement, and per-particle CTF refinement
      
      **Input**: ``Particles``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`filter` Particle filtering

      **Description**: Removal of bad particles from downstream analysis.
      
      **Input**: ``Particles``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`crosshairs` tomoDRGN (train-vae)

      **Description**: Train tomoDRGN model to learn structural heterogeneity.
      
      **Input**: ``Tomograms``
      
      **Output**: ``DRGN Model``

    .. nextpyp:: :fa:`crosshairs` tomoDRGN (analyze)

      **Description**: Analyze structural heterogeneity from tomoDRGN model.
      
      **Input**: ``DRGN Model``
      
      **Output**: ``DRGN Particles``

    .. nextpyp:: :fa:`crosshairs` tomoDRGN (analyze-volumes)

      **Description**: Volume space heterogeneity analysis from tomoDRGN model.
      
      **Input**: ``DRGN Model``
      
      **Output**: ``DRGN Particles``

    .. nextpyp:: :fa:`crosshairs` tomoDRGN (filter-star)

      **Description**: Select particle subsets from tomoDRGN clusters.
      
      **Input**: ``DRGN Particles``
      
      **Output**: ``Particles``

    .. nextpyp:: :fa:`film` Movie refinement

      **Description**: Per-particle tilt movie refinement and reconstruction, data-driven dose-weighting.
      
      **Input**: ``Particles`` or ``Frames``
      
      **Output**: ``Frames``

    .. nextpyp:: :fa:`crop` Create mask

      **Description**: Creation of shape mask.
      
      **Input**: ``Particles`` or ``Frames``
      
      **Output**: ``None``

    .. nextpyp:: :fa:`star` Post-processing

      **Description**: Masking, map sharpening and Fourier Shell Correlation (FSC) plots.
      
      **Input**: ``Particles`` or ``Frames``
      
      **Output**: ``None``


Operations
----------

Users can access block-level operations using the menu icon :fa:`bars` located at the top-right corner of each block. The following operations are supported:

* :fa:`tag` Rename block.
* :fa:`copy` Create a new block with the same input connection and parameter settings as the current block
* :fa:`edit` Open a dialog to Edit/Read block parameters. Click :bdg-primary:`Save`, :bdg-primary:`Reset` or :fa:`window-close` (to discard your changes)
* :fa:`external-link-alt` Reveal the location of the latest set of logs in the **Jobs** panel
* :fa:`location-arrow` Reveal location of files in the filesystem for the block
* :fa:`recycle` Reset state to allow re-running the block
* :fa:`eraser` Delete all files associated with the block
* :fa:`trash` Delete block. This operation cannot be undone. If a block has connections downstream, all connected blocks will be deleted (user will be prompted to confirm this operation)

Status
------

Blocks can be in one of four states (indicated by icons displayed on the top bar of each block):

* Block is up-to-date (no graphical indication)
* :fa:`star` Newly created (block was created and is ready to be executed)
* :fa:`recycle` Modified (block parameters were modified and the block needs to be updated)
* :fa:`cog fa-pulse` Running (the block is currently running)

Parameters
----------

Block parameters are specified using dialog forms. These are shown every time a new block is created or copied, or when clicking the icon :fa:`bars` and selecting the :fa:`edit` Edit option.

Jobs panel
==========

The **Jobs** panel is used to monitor the status of all jobs launched by ``nextPYP``

Jobs can be in one of four states:

* :fa:`stopwatch` Scheduled
* :fa:`cog fa-pulse` Running
* :fa:`check-circle` Completed
* :fa:`ban` Canceled
* :fa:`exclamation-triangle` Failed

Jobs are arranged hierarchically according to their dependencies, and the number of jobs in each state is continuously updated

To facilitate navigation, jobs are grouped chronologically into ``Today``, ``This Week`` and ``Older``

The arrows :fa:`angle-right` and :fa:`angle-down` are used to expand or collapse groups of jobs

Most jobs in ``nextPYP`` have three phases: *Launch*, *Split* and *Merge*. See `Compute resources <../reference/computing.html#compute-resources>`_ for more details.

.. tip::
    - A summary of currently running jobs from all projects can be found at the bottom of the **Dashboard** :fa:`tachometer-alt` page
    - Running jobs can be cancelled by clicking on the icon :fa:`ban` in the **Jobs** panel
    - Job logs can be accessed by clicking the icon :fa:`file` next to the job name in the **Jobs** panel
    - Log windows can be docked/undocked by clicking the icon :fa:`thumbtack`

Navigation
==========

Use the breadcrumb menu at the top of the page to quickly navigate to the **Dashboard** or the current **Project** page:

.. figure:: ../images/tutorial_tomo_pre_process_page.webp
  :alt: Breadcrums
