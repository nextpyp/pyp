==============================
Filter micrographs/tilt-series
==============================

**Filters** allow users to select a subset of micrographs or tilt-series for processing downstream.

Create filters
--------------

- Click inside the :bdg-secondary:`Pre-processing` block and go to the **Table** view.

- Create a new filter by providing a name and clicking :bdg-primary:`Save`.

- Add new criteria to the filter using the :bdg-primary:`+Add` button.

- Select the desired parameter from the dropdown list and set the range using the sliders.

- Use :bdg-primary:`Apply filters` to preview the results of the filter.

- You can also manually include or exclude individual micrographs or tilt-series using the **Exclude** column in the table. Shortcuts are provided to facilitate the labeling of large datasets, for example, type **x** to exclude and **c** to include an image and advance to the next or previous entry in the table.

- Once you are satisfied with the results, click :bdg-primary:`Save` to store the filter settings.

Here is a screenshot showing the filter dropdown menu for a :bdg-primary:`Particle refinement` block:

.. figure:: ../images/guide_create_new_filter.webp
    :alt: Create new filter

.. admonition:: Tips

    - You can create multiple filters by providing additional names and clicking :bdg-primary:`Save`.
    - Saved filters can be retrieved by clicking :bdg-primary:`Load` and selecting the name of the filter from the list.
    - Once a filter is loaded, it can be edited, or deleted by clicking :bdg-primary:`Delete` (this operation cannot be undone).

Apply filters
-------------

- Filters are applied to blocks downstream of the :bdg-secondary:`Pre-processing` block

- Create a new block downstream from the :bdg-secondary:`Pre-processing` block. Select the name of the desired filter from the **Filter micrographs** or **Filter tilt-series** dropdown menu, and click :bdg-primary:`Save`. When you execute the new block, only the micrographs (or tilt-series) selected by the filter will be processed.

.. figure:: ../images/guide_select_new_filter.webp
    :alt: Select filter

.. admonition:: Tips

    - Filters are applied only when first running blocks downstream of the :bdg-secondary:`Pre-processing` block. If a filter is updated or a different filter is selected, the option to ``Delete files and data`` must be selected before re-running the block (or a new block downstream of the :bdg-secondary:`Pre-processing` should be created).

    - You can experiment using different subsets of micrographs or tilt-series by creating multiple :bdg-secondary:`Particle refinement` blocks and selecting a different filter for each block.
