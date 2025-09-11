======================
Sub-tomogram averaging
======================

Sub-tomogram alignment-through-classification is a powerful strategy for *de novo* structure determination. It involves iterative 3D classification, alignment, and averaging of sub-volumes as described in  `Bartesaghi et al., 2008 <https://doi.org/10.1016/j.jsb.2008.02.008>`_. Initially, homogeneous particle groups are identified through 3D classification and subsequently averaged in 3D. The resulting class averages are then aligned to one another and combined into high signal-to-noise (SNR) references, which can be used to align individual sub-volumes. The resulting 3D models can then serve as references for high-resolution refinements using 2D projections.

Requirements
------------

- An existing :bdg-secondary:`Particle picking` block with selected particles.

Modes of operation
------------------

The sub-tomogram averaging functionality in ``nextPYP`` is provided by the :bdg-secondary:`Sub-tomogram averaging` block, which supports four primary modes of operation:

#. **Mode 0: Global averaging and iterative centering**. Computes a global average of all sub-volumes, which can then be used (optionally) as a reference to iteratively center all sub-volumes using translation-only alignment. To enhance accuracy and reduce model bias, a radially symmetrized global average can be used as the reference.

#. **Mode 1: 3D classification**. Based on the most recent set of alignments from the previous mode, sub-volumes are clustered into discrete classes, and class averages are computed.

#. **Mode 2: Class average alignment**. Class averages are aligned to each other using a user-specified reference class. The user also selects which classes to retain. After alignment, the selected classes are averaged to produce a new reference volume.

#. **Mode 3: Sub-volume alignment to reference**. Individual sub-tomograms are aligned to the reference generated in the previous step. Rotational alignment can either be global (searching the entire SO(3) space) or restricted to in-plane rotations around a pre-determined normal direction. When possible, restricting the rotation space often results in more accurate alignments.

Masks and filters for alignment and classification
--------------------------------------------------

For all modes, you can configure masking and filtering settings:

- **Masking**: Specify a radius in the x, y, and z directions and the apodization width, all in binned pixels.

- **Filtering**: Set low-pass and high-pass filter cutoffs and decay parameters, expressed as fractions of the Nyquist frequency.

Preparation
~~~~~~~~~~~

#. To access sub-volumes for averaging, generate them via the :bdg-secondary:`Particle picking` block. In the **Particle extraction** tab, set the ``Sub-volume export format`` to `3davg`, and define the desired ``Sub-tomogram size (voxels)`` and ``Sub-tomogram binning``.
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

.. note::

    To manage computational resources effectively, we recommend a sub-volume size of 64 voxels and calculating the binning accordingly. For instance, if particles are ~100 Å in diameter and the pixel size is 1 Å, using a 64-voxel box with a binning factor of 4 ensures the box is about 2.5x the particle diameter. Larger box sizes are allowed but will significantly increase computation time.

1. Global average and centering
-------------------------------

After generating the sub-volumes, configure and run the :bdg-secondary:`Sub-tomogram averaging` block:

#. Click on ``Particles`` (output of the :bdg-secondary:`Particle picking` block), then choose :bdg-primary:`Sub-tomogram averaging`. This will create a new block and show the form to enter parameters.
#. Under ``Alignments from sub-volume averaging (*.txt)``, navigate to the ``frealign`` directory from the upstream block and select the ``*_original_volumes.txt`` file.
#. Choose `mode 0 - global average and centering` as the ``Refinement mode``.
#. To use a radially symmetrized average for centering, enable ``Rotational symmetry``, set the number of centering iterations, and adjust any masking or filtering settings.
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
#. Review results in the **Global averages** tab within the :bdg-secondary:`Sub-tomogram averaging` block.

2. 3D classification
~~~~~~~~~~~~~~~~~~~~

To perform 3D classification:

#. Return to the project page and select ``Edit`` from the block menu.
#. Choose `mode 1 - classification` as the ``Refinement mode``.
#. Set the number of desired classes and configure masking or filtering as needed.
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
#. View results in the **Classes** tab of the :bdg-primary:`Sub-tomogram averaging` block.

3. Selection and alignment of class averages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To align selected class averages to a reference:

#. Return to the project page and select ``Edit`` from the block menu.
#. Choose `mode 2 - alignment of averages` as the ``Refinement mode``.
#. Specify the class selection, listing the reference class first (e.g., `5,1,3,4` aligns classes 1, 3, and 4 to class 5).
#. Set masking and filtering options as needed.
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.
#. Review aligned classes in the **Classes (aligned)** tab of the :bdg-primary:`Sub-tomogram averaging` block.

.. note::

    Steps 1-3 benefit from multithreading. Be sure to configure ``Launch, Threads`` in the **Resources** tab accordingly.

4. Alignment of sub-tomograms to reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To align all sub-volumes to the generated reference:

#. Return to the project page and select ``Edit`` from the block menu.
#. Choose `mode 3 - alignment to reference` as the ``Refinement mode``.
#. Set the parameters for rotational and translational search, along with masking and filtering options.
#. Click :bdg-primary:`Save`, :bdg-primary:`Run`, and :bdg-primary:`Start Run for 1 block`.

.. note::

    This is the most computationally intensive step. If available, the workload can be distributed across multiple nodes using a SLURM cluster.

5. Iterative refinement
~~~~~~~~~~~~~~~~~~~~~~~

#. Return to the project page and select ``Edit`` from the block menu.
#. Increase the ``Iteration number`` and repeat steps 2–4 to iteratively refine your model. 

A typical workflow might look like:

- **Iteration 1**:
    - Steps 1-4 (modes 0-3)
- **Iteration 2**:
    - Steps 2-4 (modes 1-3)
- **Iteration 3**:
    - Steps 2-4 (modes 1-3)
- *...continue as needed*

While this process can be automated, we recommend executing each step manually to improve the quality and accuracy of the resulting 3D reconstructions.