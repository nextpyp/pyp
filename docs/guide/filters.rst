==============================
Filter micrographs/tilt-series
==============================

``nextPYP`` allows users to select a subset of micrographs or tilt-series for further processing through the use of **Filters**.

Step 1: Create a filter
-----------------------

- Click inside the :badge:`Pre-processing,badge-secondary` block and go to the **Table** view.

- Create a new filter by providing a name and clicking :badge:`Save,badge-primary`.

- Add new criteria to the filter using the :badge:`+Add,badge-primary` button.

- Select the desired parameter from the dropdown list and set the range using the sliders.

- Use :badge:`Apply filters,badge-primary` to preview the results of the filter.

- You can also manually include or exclude individual micrographs or tilt-series using the **Exclude** column in the table. Shortcuts are provided to facilitate the labeling of large datasets: type **x** to exclude and **c** to include an image and advance to the next or previous entry in the table.

- Once you are satisfied with the results, click :badge:`Save,badge-primary` to save the filter settings.

.. figure:: ../images/guide_create_new_filter.webp
    :alt: Create new filter

.. tip::
    - You can create additional filters by providing a different name and clicking :badge:`Save,badge-primary`.
    - Saved filters can be retrieved by clicking :badge:`Load,badge-primary` and selecting the name of the filter from the list.
    - Once a filter is loaded, it can be edited or deleted by clicking :badge:`Delete,badge-primary` (this operation cannot be undone).

Step 2: Apply the filter
------------------------

- Create a new refinement block downstream from the :badge:`Pre-processing,badge-secondary` block. Select the name of the desired filter from the **Filter micrographs** or **Filter tilt-series** dropdown menu, and click :badge:`Save,badge-primary`. When you execute the :badge:`Particle refinement,badge-secondary` block, only the micrographs (or tilt-series) selected by the filter will be used for the processing downstream.

.. figure:: ../images/guide_select_new_filter.webp
    :alt: Select filter

.. tip::
    - Filters do not work if you have previously executed the :badge:`Particle refinement,badge-secondary` block without using a filter or using a different filter (you need to create a new :badge:`Particle refinement,badge-secondary` block in this case).

    - You can experiment using different subsets of micrographs or tilt-series by creating multiple :badge:`Particle refinement,badge-secondary` blocks and selecting a different filter for each block.

.. seealso::

    * :doc:`2D particle picking<picking2d>`
    * :doc:`3D particle picking<picking3d>`
    * :doc:`Pattern mining<milopyp>`
    * :doc:`Visualization in ChimeraX/ArtiaX<chimerax_artiax>`
    * :doc:`Overview<overview>`