=======
License
=======

BSD 3-Clause License
--------------------

.. code-block:: bash

   Copyright (c) 2025, Alberto Bartesaghi

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Dependencies
------------

``nextPYP`` uses functionality from `IMOD <https://bio3d.colorado.edu/imod/>`_, `cisTEM <https://cistem.org/>`_, `CTFFIND4 <https://grigoriefflab.umassmed.edu/ctffind4>`_, `AreTomo2 <https://github.com/czimaginginstitute/AreTomo2>`_, `MotionCor3 <https://github.com/czimaginginstitute/MotionCor3>`_, `Topaz <https://github.com/tbepler/topaz>`_, `IsoNet <https://github.com/IsoNet-cryoET/IsoNet>`_, `cryoCARE <https://github.com/juglab/cryoCARE_pip>`_, `MemBrain-Seg <https://github.com/teamtomo/membrain-seg>`_, `tomoDRGN <https://github.com/bpowell122/tomodrgn>`_, and `pytom-match-pick <https://github.com/SBC-Utrecht/pytom-match-pick>`_.


The corresponding licenses are reproduced below.

cisTEM and CTFFIND4
===================

.. code-block:: bash

   Copyright © 2023 Howard Hughes Medical Institute

   Redistribution and use in source and binary forms, with or without modification,
   are permitted provided that the following conditions are met:

   - Redistributions of source code must retain the above copyright notice, this 
   list of conditions and the following disclaimer.
   - Redistributions in binary form must reproduce the above copyright notice, 
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
   - Neither the name of HHMI nor the names of its contributors may be used to 
   endorse or promote products derived from this software without specific 
   prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND 
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
   IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
   OF THE POSSIBILITY OF SUCH DAMAGE.

IMOD
====

.. code-block:: bash

   IMOD Version 4.11                      Copyright Notice and Credits
   --------------------------------------------------------------------------
   Except as noted, ALL SOFTWARE LISTED IS Copyright (C) 1994-2020
   by the Regents of the University of Colorado.

   All portions of IMOD, except as noted below, are open source under the
   General Public License (GPL) version 2.0.  A copy of this license is in the
   file GPL.txt.  (All license files referred to here are in the directory
   'licenses' in the binary distribution or 'dist' in the source code.)  The 
   libraries written entirely in C/C++ are released under the Lesser GPL (see
   LGPL.txt).  Software may be modified and redistributed under the terms of
   these licenses.  The source can be found at
   http://bio3d.colorado.edu/imod/nightlyBuild
   and
   http://bio3d.colorado.edu/imod/openSource

   THIS SOFTWARE AND/OR DOCUMENTATION IS PROVIDED WITH NO WARRANTY,
   EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, WARRANTY OF
   MERCHANTABILITY AND WARRANTY OF FITNESS FOR A PARTICULAR PURPOSE.

   Programs were written by David Mastronarde, James Kremer, Rick
   Gaudette, Sue Held, Quanren Xiong, and John Heumann at the University 
   of Colorado, some based on work and libraries from the MRC.  We thank David
   Agard and Hans Chen for the original MRC source code, Wah Chiu for a Unix
   version of low-level MRC file routines used in the FORTRAN parts, and Chuck
   Robertson for much work on the port to Linux.

   The program nad_eed_3d by Achilleas Frangakis and Reiner Hegerl is
   copyright Max-Planck-Institut for Biochemistry, Martinsried, Germany.

   The LSQR routine by Michael Saunders is used in some programs.  IMOD uses a
   C version of LSQR and some BLAS routines from the BCLS package of Michael
   Friedlander, which are covered by the Lesser GPL (see LGPL.txt).

   IMOD uses a subset of the LAPACK routines developed at Univ. of Tennessee,
   Univ. of California Berkeley, NAG Ltd., Courant Institute, Argonne National
   Lab, and Rice University.  They are copyrighted by the University of
   Tennessee and covered by a BSD-style license (see LAPACK.txt).  IMOD also
   uses a subset of the BLAS routines, developed by: Jack Dongarra, Argonne
   National Lab; Jeremy Du Croz, NAG Ltd.; Iain Duff, AERE Harwell; Richard
   Hanson, Sandia National Labs; and Sven Hammarling, NAG Ltd.

   The program imodwincpu was adapted from code obtained from
   www.codeproject.com and written by Dudi Abramov.  It is excluded from the
   GPL license and is covered exclusively by the Code Project Open License (see
   CPOL.html). 

   Routines for computing some statistical functions were adapted from
   S. J. Zhang and J. M. Jin, Computation of Special Functions. New York: John
   Wiley & Sons, 1996 and are copyrighted by the authors and publisher.

   Andrew Noske (originally at the University of Queensland, Australia, then at
   the University of California, San Diego) contributed the beadhelper,
   drawingtools, interpolator, namewizard, and stereology plugins.  See the help
   displayed by those plugins for acknowledgements of funding for his work.

   Jane Ding at California Institute of Technology contributed the Grab with Note
   plugin.

   The isosurface display in 3dmod uses contouring and surface smoothing
   modules from Chimera, developed at the Resource for Biocomputing,
   Visualization, and Informatics at the University of California, San
   Francisco, supported by NIH/NCRR grant P41-RR01081.

   RAPTOR was developed by Fernando Amat, Farshid Moussavi, and Mark Horowitz
   at Stanford University and is copyrighted by them.  It is covered by the
   license in RAPTORlicense.txt.  It uses three libraries, parts of which are
   included in the IMOD source code distribution:
   OpenCV (http://sourceforge.net/projects/opencvlibrary/), 
   covered by the license in OpenCV.txt, 
   SuiteSparse (http://www.cise.ufl.edu/research/sparse/SuiteSparse/), 
   covered by the licenses in CSparse.txt and LGPL.txt, and 
   Stair Vision Library (http://sourceforge.net/projects/stairvision/), 
   covered by the license in StairVision.txt.

   The warping library uses modules from Pavel Sakov's 'nn' package, which are
   copyrighted by Sakov and CSIRO, and covered by the license in nn.txt.  The
   library also uses code from Ken Clarkson's 'hull' program, which is
   copyrighted by AT&T and covered by the license in hull.txt.

   Routines for solving 3x3 eigenvectors are copyrighted by Joachim Kopp and
   covered by the LGPL license.

   The ctffind library was adapted from the ctffind program of Alexis Rohou and
   Nikolaus Grigorieff, which is Copyright (c) 2018, Howard Hughes Medical
   Institute, and is covered by the Janelia Research Campus Software License
   1.2.

   Mauro Maiorca, at the Biochemistry & Molecular Biology Department, Bio21
   Institute, University of Melbourne, Australia, contributed the preNAD and
   preNID programs.  His work was supported by funding from the Australian
   Research Council and the National Health and Medical Research Council.  preNAD
   and preNID use recursive line filter routines from Gregoire Malandain, covered
   by version 3 of the GPL (see GPL-3.0.txt).

   IMOD uses TIFF libraries which are Copyright (c) 1988-1997 Sam Leffler
   and Copyright (c) 1991-1997 Silicon Graphics, Inc. (see TIFF.txt).

   Because IMOD uses the libjpeg library, this software is based in part on the
   work of the Independent JPEG Group.  IMOD also uses the zlib library, which is
   Copyright 1995-2010 by Jean-loup Gailly and Mark Adler.

   IMOD may use FFTW libraries which are Copyright (c) 1997--1999 Massachusetts
   Institute of Technology, written by Matteo Frigo and Steven G. Johnson, and
   covered by version 2 of the GPL.

   IMOD uses HDF5 libraries which are Copyright 1998-2006 by the Board of
   Trustees of the University of Illinois and Copyright 2006-2014 by The HDF
   Group and covered by the license in HDF5.txt.

   The module gcvspl.c is based on an f2c translation of gcvspl.f, which was
   obtained from http://www.netlib.org.  gcvspl.f was written by H.J. Woltring
   based on routines in Lyche et al. (1983) and other sources as documented in
   gcvspl.c.

   IMOD includes a copy of the Mini-XML library which is Copyright 2003-2016 by
   Michael R. Sweet and is covered by the modified Library GPL in Mini-XML.txt

   This work is supported by NIH/NIGMS grant GM125074 to David Mastronarde.

   Contact:  mast at colorado dot edu
      www:  http://bio3d.colorado.edu/imod/index.html
      University of Colorado, Dept. of MCD Biology, 347 UCB, Boulder, CO 80309

AreTomo2 and MotionCor3
=======================

.. code-block:: bash

   Copyright 2023 Chan Zuckerberg Institute for Advanced Biological Imaging
   Redistribution and use in source and binary forms, with or without modification,
   are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list
      of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this
      list of conditions and the following disclaimer in the documentation and/or
      other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its contributors may
      be used to endorse or promote products derived from this software without specific
      prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
   SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
   TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
   DAMAGE.

Topaz
=====

Topaz is distributed under the `GNU General Public License v3.0 <https://github.com/tbepler/topaz/blob/master/LICENSE>`__.


IsoNet
======

.. code-block:: bash

    MIT License

    Copyright (c) 2021 Yun-Tao Liu, Heng Zhang, Hui Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

cryoCARE
========

.. code-block:: bash

    BSD 3-Clause License

    Copyright (c) 2020, juglab
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

MemBrain-Seg
============

.. code-block:: bash
  
    BSD License

    Copyright (c) 2023, Lorenz Lamm
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


tomoDRGN
========

tomoDRGN is distributed under the `GNU General Public License v3.0 <https://github.com/bpowell122/tomodrgn/blob/master/LICENSE.txt>`__.


pytom-match-pick
================

pytom-match-pick is distributed under the `GNU General Public License v2.0 <https://github.com/SBC-Utrecht/pytom-match-pick/blob/main/LICENSE>`__.
