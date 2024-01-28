========
Overview
========

``nextPYP``'s graphical user interface (GUI) has the following components:

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
  - Blocks can be moved around or organized within the canvas
  - Multiple blocks can be selected and moved as a set.

Block connectivity
------------------

Each block in ``nextPYP`` has inputs and outputs of specific types. Two blocks can be connected only if their input/outputs types are compatible. The program does not allow for incompatible block connections.

Block types
-----------

Single-particle projects and tomography projects will have slightly differing block types. A list of block types and their corresponding inputs and outputs is given below:

.. tabbed:: Single-particle

  .. dropdown:: :fa:`layer-group fa-2x` Data import
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Import movie frames or micrographs in multiple formats (MRC, DM4, TIF, and EER).
    :Input: None
    :Output: ``Movies``

  .. dropdown:: :fa:`chart-bar fa-2x` Pre-processing
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:
 
    :Description: Movie frame alignment, CTF estimation, and particle picking.
    :Input: ``Movies``
    :Output: ``Particles``

  .. dropdown:: :fa:`dot-circle fa-2x` Particle refinement
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Particle alignment, classification and per-particle CTF refinement.
    :Input: ``Particles``
    :Output: ``Particles``

  .. dropdown:: :fa:`filter fa-2x` Particle filtering
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Removal of bad particles from downstream analysis.
    :Input: ``Particles``
    :Output: ``Particles``

  .. dropdown:: :fa:`film fa-2x` Movie refinement
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Per-particle movie frame alignment and dose weighting.
    :Input: ``Particles`` or ``Frames``
    :Output: ``Frames``

  .. dropdown:: :fa:`crop fa-2x` Create mask
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Creation of shape masks.
    :Input: ``Particles`` or ``Frames``
    :Output: ``None``

  .. dropdown:: :fa:`star fa-2x` Post-processing
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Masking, map sharpening, and Fourier Shell Correlation (FSC) plots.
    :Input: ``Particles`` or ``Frames``
    :Output: ``None``

.. tabbed:: Tomography

  .. dropdown::  :fa:`layer-group fa-2x` Data import
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Decription: Import raw tilt-series data (with or without frames) in MRC or TIF format.
    :Input: None
    :Output: ``Tilt-series``

  .. dropdown:: :fa:`chart-bar fa-2x` Pre-processing
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Frame and tilt-series alignment, tomogram reconstruction, CTF estimation, and particle picking.
    :Input: ``Tilt-series``
    :Output: ``Particles``

  .. dropdown:: :fa:`dot-circle fa-2x` Particle refinement
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Constrained particle alignment and classification, region-based refinement, and per-particle CTF refinement
    :Input: ``Particles``
    :Output: ``Particles``

  .. dropdown:: :fa:`filter fa-2x` Particle filtering
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Removal of bad particles from downstream analysis.
    :Input: ``Particles``
    :Output: ``Particles``

  .. dropdown:: :fa:`film fa-2x fa-2x fa-2x` Movie refinement
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Per-particle tilt movie refinement and reconstruction, data-driven dose-weighting.
    :Input: ``Particles``or ``Frames``
    :Output: ``Frames``

  .. dropdown:: :fa:`crop fa-2x` Create mask
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Creation of shape mask.
    :Input: ``Particles``or ``Frames``
    :Output: ``None``

  .. dropdown:: :fa:`star fa-2x` Post-processing
    :container: + shadow
    :title: bg-primary text-white text-left font-weight-bold
    :open:

    :Description: Masking, map sharpening and Fourier Shell Correlation (FSC) plots.
    :Input: ``Particles`` or ``Frames``
    :Output: ``None``


Block operations
----------------

Users can access block-level operations using the menu icon :fa:`bars,text-primary` located at the top-right corner of each block. The following operations are supported:

* :fa:`tag text-primary` Rename block.
* :fa:`copy text-primary` Create a new block with the same input connection and parameter settings as the current block.
* :fa:`edit text-primary` Open a dialog to Edit/Read block parameters. Click :badge:`Save, badge-primary`, :badge:`Reset, badge-primary` or close the dialog by clicking the icon :fa:`window-close` to discard your changes.
* :fa:`external-link-alt text-primary` Reveal the location of the latest set of logs for the block in the **Jobs** panel.
* :fa:`location-arrow text-primary` Reveal location of files in the filesystem for the block.
* :fa:`recycle text-primary` Reset state to allow re-running the block.
* :fa:`eraser text-primary` Delete all files associated with the block.
* :fa:`trash text-primary` Delete block. This operation cannot be undone. If a block has connections downstream, all connected blocks will be deleted (user is required to confirm this operation).

Block status
------------

Blocks can be in one of three states (indicated by icons displayed on the top bar of each block):

* Block is up-to-date (no graphical indication)
* :fa:`recycle text-primary` Modified (parameters were modified and the block is not up-to-date)
* :fa:`cog fa-pulse text-primary` Running (the block is currently running)

Block parameters
----------------

Block parameters are specified using dialog forms. These are shown every time a new block is created or copied, or when clicking the icon :fa:`bars,text-primary` and selecting the :fa:`edit,text-primary` Edit option.

Jobs panel
==========

The **Jobs** panel is used to monitor the status of all SLURM jobs launched by ``nextPYP``

Jobs can be in one of four states:

* :fa:`stopwatch` Scheduled
* :fa:`cog fa-pulse` Running
* :fa:`check-circle` Completed
* :fa:`exclamation-triangle` Failed

Jobs are arranged hierarchically according to their dependencies, and the number of jobs in each state is updated continuously.

For simplicity, jobs are grouped chronologically into ``Today``, ``This Week`` and ``Older``.

The arrows :fa:`angle-right` and :fa:`angle-down` are used to expand or collapse each group.

The three job phases *Launch*, *Split* and *Merge* within each run are organized according to their dependencies.

.. tip::
    - A summary of currently running jobs from all projects in ``nextPYP`` can be found in the **Dashboard** :fa:`tachometer-alt` page
    - Running jobs can be cancelled by clicking on the icon :fa:`ban,text-danger`
    - Job logs can be accessed by clicking the icon :fa:`file,text-primary` to the right of the job name
    - The log window can be docked/undocked by clicking the icon :fa:`thumbtack,text-primary`

Navigation
==========

Use the breadcrumb menu at the top of the page to navigate to the **Dashboard** or the current **Project**

.. figure:: ../images/tutorial_tomo_pre_process_page.webp
  :alt: Breadcrums

.. tip::
    Some pages in ``nextPYP`` can be bookmarked and saved for later reference

.. seealso::

    * :doc:`Particle picking<picking>`
    * :doc:`Neural-network picking<neural_network>`
    * :doc:`Filter micrographs/tilt-series<filters>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`