==============================
Filter micrographs/tilt-series
==============================

``nextPYP`` allows users to select a subset of micrographs or tilt-series for further processing through the use of **Filters**.

Create a filter
---------------

- Click inside the :bdg-secondary:`Pre-processing` block and go to the **Table** view.

- Create a new filter by providing a name and clicking :bdg-primary:`Save`.

- Add new criteria to the filter using the :bdg-primary:`+Add` button.

- Select the desired parameter from the dropdown list and set the range using the sliders.

- Use :bdg-primary:`Apply filters` to preview the results of the filter.

- You can also manually include or exclude individual micrographs or tilt-series using the **Exclude** column in the table. Shortcuts are provided to facilitate the labeling of large datasets: type **x** to exclude and **c** to include an image and advance to the next or previous entry in the table.

- Once you are satisfied with the results, click :bdg-primary:`Save` to save the filter settings.

.. figure:: ../images/guide_create_new_filter.webp
    :alt: Create new filter

.. tip::
    - You can create additional filters by providing a different name and clicking :bdg-primary:`Save`.
    - Saved filters can be retrieved by clicking :bdg-primary:`Load` and selecting the name of the filter from the list.
    - Once a filter is loaded, it can be edited, or deleted by clicking :bdg-primary:`Delete` (this operation cannot be undone).

Apply the filter
----------------

- Create a new refinement block downstream from the :bdg-secondary:`Pre-processing` block. Select the name of the desired filter from the **Filter micrographs** or **Filter tilt-series** dropdown menu, and click :bdg-primary:`Save`. When you execute the :bdg-secondary:`Particle refinement` block, only the micrographs (or tilt-series) selected by the filter will be used for the processing downstream.

.. figure:: ../images/guide_select_new_filter.webp
    :alt: Select filter

.. tip::
    - Filters do not work if you have previously executed the :bdg-secondary:`Particle refinement` block without using a filter or using a different filter (you need to create a new :bdg-secondary:`Particle refinement` block in this case).

    - You can experiment using different subsets of micrographs or tilt-series by creating multiple :bdg-secondary:`Particle refinement` blocks and selecting a different filter for each block.
