A scalable platform for cryo-EM/ET image analysis
-------------------------------------------------

`nextPYP <https://nextpyp.app/>`_ is a comprehensive platform for single-particle cryo-EM/ET image analysis developed and maintained by the `Bartesaghi Lab <http://cryoem.cs.duke.edu>`_ at `Duke University <http://www.duke.edu>`_.

By combining established tools in the field with `methods <https://cryoem.cs.duke.edu/research/methods/>`_ developed in-house, ``nextPYP`` produces state-of-the-art results while being easy-to-use, portable, and scalable to datasets with thousands of micrographs or tomograms.

Main features
-------------
- **Ease-of-use**: portable, user-friendly, and fully featured web-based GUI
- **Scalability**: parallel processing and small storage footprint (no sub-volumes or particle stacks saved)
- **End-to-end pipeline**: runs all steps required to convert raw cryo-EM/ET data into 3D structures
- **Pattern mining**: deep learning-based cellular exploration and particle localization (MiLoPYP)
- **Particle picking suite**: size-based, geometry-based and neural network-based picking methods
- **Constrained refinement**: multi-mode constrained refinement and classification of particle projections
- **High-resolution refinement**: per-particle CTF and frame refinement, and self-tuning exposure weighting
- **On-the-fly processing**: high-throughput analysis of micrographs/tilt-series during data collection
- **Interoperability**: import/export metadata in multiple formats to interface with external programs

See the :doc:`Changelog<changelog>` for the latest features.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. panels::
   :body: bg-primary text-centered text-white font-weight-bold

   :fa:`cog fa-4x text-white`

   +++

   .. link-button:: install/install-web
      :type: ref
      :text: Installation
      :classes: btn-outline-primary btn-block stretched-link

   ---

   :fa:`book-open fa-4x text-white`

   +++

   .. link-button:: tutorials
      :type: ref
      :text: Tutorials
      :classes: btn-outline-primary btn-block stretched-link

   ---

   :fa:`user fa-4x text-white`

   +++

   .. link-button:: guide
      :type: ref
      :text: User guide
      :classes: btn-outline-primary btn-block stretched-link

   ---

   :fa:`info fa-4x text-white`

   +++

   .. link-button:: about
      :type: ref
      :text: About
      :classes: btn-outline-primary btn-block stretched-link

.. admonition:: Need help?

   Visit ``nextPYP``'s `discussion board <https://github.com/orgs/nextpyp/discussions>`_ to post questions and follow discussions

.. admonition:: Citing ``nextPYP`` and ``MiLoPYP``

  Liu, HF., Zhou, Y., Huang, Q., Piland, J., Jin, W., Mandel, J., Du, X., Martin, J., Bartesaghi, A., `nextPYP: a comprehensive and scalable platform for characterizing protein variability in-situ using single-particle cryo-electron tomography <https://www.nature.com/articles/s41592-023-02045-0>`_. Nature Methods, 20:1909–1919 (2023).

  Huang, Q., Zhou, Y., Bartesaghi, A. `MiLoPYP: self-supervised molecular pattern mining and particle localization in situ <https://www.nature.com/articles/s41592-024-02403-6>`_, Nature Methods, in press (2024).