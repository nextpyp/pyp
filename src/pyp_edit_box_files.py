#!/usr/bin/env python

import glob
import os
import sys
from pathlib import Path

import numpy
import pandas as pd

from pyp.inout.metadata import frealign_parfile, pyp_metadata
from pyp.inout.utils.pyp_edit_box_files import (
    produce_box_files,
    produce_boxx_files_fast,
)
from pyp.system import project_params
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


if __name__ == "__main__":

    # This file does several functions:
    # 1. Produce .box files from manual particle picking using IMOD's mod file (mod2box).
    # 2. Produce image and model file from set of boxx files (box2mod)
    # 3. Generate clean set of particles based on 2D classification (imagic2box,eman2box).

    # Protocol for generating clean set of .boxx files:
    # 1. pick particles as usual
    # 2. run IMAGIC/EMAN 2D classification on CTF corrected stack (make sure you use outliers=0 if using IMAGIC)
    # 3a. Manually select BAD classes and save in IMOD model (if using IMAGIC)
    # 3b. Manually select GOOD classes and save in file using e2evalparticles.py (if using EMAN)
    # 4. Run this program to label particles in bad classes (information saved in boxx files)
    # 5. Re-pick particles (this will use the new boxx files that ignore the bad particles).

    args = project_params.parse_arguments("editboxfiles")

    parfile_extension = Path(args.parfile).suffix

    parameters = project_params.load_pyp_parameters()

    if parfile_extension == ".par":
        logger.info("Recognized parameter file as FREALIGN format.")
        mode = "frealign2box"
    elif parfile_extension == ".star":
        logger.info("Recognized parameter file as RELION format.")
        mode = "relion2box"
    elif parfile_extension == ".mod":
        logger.info("Recognized parameter file as IMOD format.")
        mode = "imod2box"
    else:
        logger.info("Recognized parameter file as EMAN format.")
        mode = "eman2box"

    # this mode is used when manually picking particles from a dataset
    if mode == "mod2box":

        if (
            not args.modfile
            or not args.listfile
            or not args.binning
            or not args.boxsize
        ):

            logger.info(
                "Parameters -modfile, -listfile, -binning and -boxsize are required."
            )

        else:

            produce_box_files(args.modfile, args.listfile, args.binning, args.boxsize)

    elif mode == "box2mod":

        # produce pair of .mod and .txt files from set of .boxx files

        if not args.modfile or not args.binning or not args.listfile:

            logger.info("Parameters -modfile and -binning are required.")

        else:
            # create images file
            f = open(args.modfile.replace(".mod", ".txt"), "wb")
            f.write("IMOD image list \nVERSION 1 \n")

            # temporary file to save coordinates
            p = open("coordinates.txt", "wb")
            count = sizey = 0

            # for all .boxx files in current directory
            for name in open(args.listfile, "rb").read().split("\n"):

                # determine image dimensions
                if sizey == 0:
                    sizey = Image.open(name + ".jpg").size[1] * args.binning

                # read particle coorinates
                if os.path.isfile("{}.boxx".format(name)):
                    f.write("IMAGE ./" + name + ".jpg\n")
                    boxxs = numpy.loadtxt("{}.boxx".format(name), ndmin=2)
                    boxs = boxxs[boxxs[:, 4] + boxxs[:, 5] == 2]
                    for i in range(boxs.shape[0]):
                        p.write(
                            "%i\t%i\t%i\n"
                            % (
                                (boxs[i, 0] + boxs[i, 2] / 2) / args.binning,
                                (sizey - boxs[i, 1] - boxs[i, 3] / 2) / args.binning,
                                count,
                            )
                        )
                    count += 1
            f.close()
            p.close()

            # convert to IMOD model
            command = (
                "%s/bin/point2model -zero -scat -planar -CircleSize 10 -color 255,0,0 -input %s -output %s"
                % (get_imod_path(), "coordinates.txt", args.modfile)
            )
            run_shell_command(command)
            os.remove("coordinates.txt")

    # this mode is used to elimiinate particles in bad classes using IMAGIC
    elif mode == "imagic2box":

        if not args.modfile or not args.listfile:

            logger.info("Parameters -modfile and -listfile are required.")

        else:
            global_indexes_to_remove = particle_selection_by_imagic_classification(
                args.modfile, args.listfile
            )

            global_indexes_to_keep = []
            [
                global_indexes_to_keep.append(i)
                for i in range(args.particles)
                if i not in global_indexes_to_keep
            ]

            # logger.info("Particles to remove = %i", len(global_indexes_to_remove))
            # logger.info("Particles to keep = %i", len(global_indexes_to_keep))

            produce_boxx_files(global_indexes_to_remove)
            # produce_boxx_files( global_indexes_to_keep )

    # this mode is used to elimiinate particles in bad classes using EMAN
    elif mode == "eman2box":

        if not args.parfile or not args.extract_cls:

            logger.error("Parameter -parfile and -extract_cls are required.")

        else:
            # total number of particles

            # assume indexes produced by e2evalparticles.py are already in base-0 mode and file contains GOOD particles
            f = open(args.parfile)
            A = numpy.array(f.read().split(), dtype=int).tolist()
            f.close()

            particles = 0
            parameters = project_params.load_pyp_parameters()
            micrographs = "{}.micrographs".format(parameters["data_set"])
            inputlist = [line.strip() for line in open(micrographs, "rb")]
            for name in inputlist:
                boxxfile = "box/{}.boxx".format(name)
                if os.path.exists(boxxfile):
                    boxx = numpy.loadtxt(boxxfile, ndmin=2)
                    particles += numpy.sum(
                        numpy.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= args.extract_cls - 1
                        )
                    )
            logger.info(
                "Particles that survived previous classification = %d", particles
            )

            # check consistency between indexes and current number of particles
            if max(A) > particles - 1:
                logger.error(
                    "Particle index greater than number of particles? %d > %d",
                    max(A),
                    particles,
                )
                sys.exit()

            # create list of BAD particles
            global_indexes_to_remove = []
            [global_indexes_to_remove.append(i) for i in range(particles) if i not in A]

            logger.info(
                "Particles to remove in classification pass %d = %d",
                args.extract_cls,
                len(global_indexes_to_remove),
            )
            logger.info(
                "Particles left after classification pass %d = %d",
                args.extract_cls,
                particles - len(global_indexes_to_remove),
            )

            # produce_boxx_files( global_indexes_to_remove, args.extract_cls )

            if not args.debug:

                # indexes are in base-0
                if True:
                    # produce_boxx_files( global_indexes_to_remove, args.extract_cls, args.shifts )
                    if args.shifts:
                        produce_boxx_files_fast(
                            global_indexes_to_remove, args.extract_cls, input[:, 4:6]
                        )
                    else:
                        produce_boxx_files_fast(
                            global_indexes_to_remove, args.extract_cls
                        )

    elif mode == "frealign2box":

        # select particles based on Frealign parameter files

        """
        XD's understanding of mode == 'frealign2box
        1. Load specified parfile
        2. If the threshold is > 1 or <= 0, use the threshold as the cutoff score
            Else, use threshold as the fraction of particles to take, and calculate corresponding cutoff score
        3. If there are multipel classes, combine classes
        4. Filter out particles based on score and occupancy from the par file
        5. Run produce_boxx_files_fast to update boxx files
        6. Update parfile and write to _clean.par
        """

        if not args.parfile:
            logger.error("Parameter -parfile is required.")
            sys.exit()

        else:
            if not args.extract_cls > 0:
                logger.error("-extract_cls must be greater than 0.")
                sys.exit()

            # load .par open(s) into multidimensional array
            par_files = glob.glob(str(args.parfile))

            if len(par_files) == 0:
                logger.error("{} not found.".format(args.parfile))
                sys.exit(0)
            input = numpy.array([])
            for p in sorted(par_files):
                logger.info("Reading %s", p)
                current = numpy.array(
                    [line.split() for line in open(p) if not line.startswith("C")],
                    dtype=float,
                )
                if input.size > 0:
                    input = numpy.dstack((input, current))
                else:
                    input = current

            # figure out in which field the phase residuals / scores are
            if not args.phases:
                field = 14
            elif args.frealignx:
                field = 15
            else:
                field = 11

            # figure out phase residual / score threshold
            if float(args.threshold) > 1 or float(args.threshold) <= 0:
                thresh = float(args.threshold)
            else:
                if not args.phases:
                    thresh = input[input[:, field].argsort()][
                        -int(float(args.threshold) * input.shape[0]) + 1, field
                    ]
                else:
                    thresh = input[input[:, field].argsort()][
                        int(float(args.threshold) * input.shape[0]) - 1, field
                    ]

            logger.info("Using phase residual threshold of %0.8f", thresh)

            # combine selected classes
            if input.ndim > 2:
                classes = numpy.array(sorted(args.classes.split(",")), dtype=int)
                combined = input[:, :, classes[0]]
                for p in classes[1:]:
                    condition = numpy.where(
                        combined[:, field] > input[:, field, p], 1, 0
                    )
                    select = numpy.array(condition).reshape(-1, 1) * numpy.ones(
                        ([1, input.shape[1]])
                    )
                    combined = numpy.multiply(select, combined) + numpy.multiply(
                        (1 - select), input[:, :, p]
                    )
                input = combined

            # select ignored particles
            if not args.phases:
                newinput = input[
                    numpy.logical_or(
                        input[:, field] < thresh, input[:, 11] < args.occupancy
                    )
                ]
                newinput_keep = input[
                    numpy.logical_not(
                        numpy.logical_or(
                            input[:, field] < thresh, input[:, 11] < args.occupancy
                        )
                    )
                ]
            else:
                newinput = input[input[:, field] >= thresh]
                newinput_keep = input[numpy.logical_not(input[:, field] >= thresh)]
            global_indexes_to_remove = (newinput[:, 0] - 1).astype("int").tolist()
            global_indexes_to_keep = (newinput_keep[:, 0] - 1).astype("int").tolist()

            # create list of GOOD particles
            if False:
                global_indexes_to_keep = []
                [
                    global_indexes_to_keep.append(i)
                    for i in range(input.shape[0])
                    if i not in global_indexes_to_remove
                ]
            else:
                # fast way
                global_indexes_to_keep = list(
                    set(numpy.arange(0, input.shape[0]).tolist())
                    - set(global_indexes_to_remove)
                )

            logger.info(f"Particles to remove = {len(global_indexes_to_remove):,}")
            logger.info(f"Particles to keep = {len(global_indexes_to_keep):,}")

            if not args.debug:

                # indexes are in base-0
                if True:
                    # produce_boxx_files( global_indexes_to_remove, args.extract_cls, args.shifts )
                    if args.shifts:
                        boxxdbase = produce_boxx_files_fast(
                            global_indexes_to_remove, args.extract_cls, input[:, 4:6]
                        )
                    else:
                        boxxdbase = produce_boxx_files_fast(
                            global_indexes_to_remove, args.extract_cls
                        )
                    
                    # updating metadata is necessary if track the metadata only in the future
                    if args.metadatapth and os.path.isdir(args.metadatapth):
                        is_spr = True 
                        micrographs = "{}.micrographs".format(parameters["data_set"])
                        inputlist = [line.strip() for line in open(micrographs, "r")]
                        current_dir = os.getcwd()
                        os.chdir(args.metadatapth)
                        box_header = ["x", "y", "Xsize", "Ysize", "inside", "selection"]
                        for name in inputlist:
                            metadata = pyp_metadata.LocalMetadata(name + ".pkl", is_spr=is_spr)
                            boxxdata = boxxdbase[name]
                            df = pd.DataFrame(boxxdata, columns=box_header)
                            if "box" in metadata.data.keys():
                                metadata.updateData({'box':df})    
                                metadata.write()
                        os.chdir(current_dir)

                # produce corresponding .par file
                if not args.phases:
                    # newinput = input[ input[:,field] > thresh ]
                    newinput = input[
                        numpy.logical_and(
                            input[:, field] >= thresh, input[:, 11] >= args.occupancy
                        )
                    ]
                    # newinput = input[ numpy.logical_and( input[:,field] <= thresh, input[:,11] <= args.occupancy ) ]
                else:
                    newinput = input[input[:, field] < thresh]
                newinput[:, 0] = list(range(newinput.shape[0]))
                newinput[:, 0] += 1

                if newinput.shape[0] != len(global_indexes_to_keep):
                    logger.error(
                        "Number of clean particles does not match number of particles to keep: {0} != {1}".format(
                            newinput.shape[0], len(global_indexes_to_keep)
                        )
                    )
                    sys.exit()

                # re-number films
                current_film = newinput[0, 7]
                new_film_number = 0
                for i in range(newinput.shape[0]):
                    if newinput[i, 7] != current_film:
                        current_film = newinput[i, 7]
                        new_film_number += 1
                    newinput[i, 7] = new_film_number

                # apply binning to image shifts (FREALIGN 9 measures shifts in Angstroms so we don't need this)
                if args.phases and args.binning > 1 and args.binning < 8:
                    newinput[:, 4:6] *= args.binning

                # set occupancy to 100
                if not args.phases:
                    newinput[:, 11] = 100.0

                outputparfile = (
                    str(args.parfile).replace(".par", "_clean.par").replace("?", "1")
                )
                f = open(outputparfile, "w")
                
                #  use standard header   
                if args.frealignx:
                    par_header = frealign_parfile.EXTENDED_FREALIGNX_PAR_HEADER
                else:
                    par_header = frealign_parfile.EXTENDED_NEW_PAR_HEADER
                header = "".join(par_header)
                f.write(header)

                if args.shifts:
                    parameters = project_params.load_pyp_parameters()
                    pixel = float(parameters["scope_pixel"]) * float(
                        parameters["data_bin"]
                    )
                    newinput[:, 4:6] -= numpy.round(newinput[:, 4:6] / pixel) * pixel

                for i in range(newinput.shape[0]):
                    if not args.phases:
                        if False:
                            f.write(
                                frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE
                                % tuple(newinput[i, :17])
                            )
                        elif args.frealignx:
                            f.write(
                                frealign_parfile.EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE
                                % tuple(newinput[i, :])
                            )
                        else:
                            f.write(
                                frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE
                                % tuple(newinput[i, :])
                            )

                    else:
                        f.write(
                            frealign_parfile.CCLIN_PAR_STRING_TEMPLATE
                            % tuple(newinput[i, :13])
                        )
                    f.write("\n")

                # TODO: add FSC statistics if needed 

                f.close()

    elif mode == "relion2box":

        if not args.parfile:

            logger.error("Parameter -parfile needed.")

        # read data star file
        input = numpy.array(
            [line.split() for line in open(args.parfile) if ".mrcs" in line]
        )

        # figure out RELION column numbers from star file header
        rlnClassNumber = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnClassNumber" in line
        ][0] - 1
        rlnVoltage = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnVoltage" in line
        ][0] - 1
        rlnDefocusU = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnDefocusU" in line
        ][0] - 1
        rlnDefocusV = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnDefocusV" in line
        ][0] - 1
        rlnDefocusAngle = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnDefocusAngle" in line
        ][0] - 1
        rlnAngleRot = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnAngleRot" in line
        ][0] - 1
        rlnAngleTilt = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnAngleTilt" in line
        ][0] - 1
        rlnAnglePsi = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnAnglePsi" in line
        ][0] - 1
        rlnOriginX = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnOriginX" in line
        ][0] - 1
        rlnOriginY = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnOriginY" in line
        ][0] - 1
        rlnGroupNumber = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnGroupNumber" in line
        ][0] - 1
        rlnImageName = [
            int(line.split()[1].replace("#", ""))
            for line in open(args.parfile)
            if "_rlnImageName" in line
        ][0] - 1

        class_file = args.parfile.replace("_data", "_model")

        if class_file != args.parfile:

            rlnAccuracyRotations = [
                int(line.split()[1].replace("#", ""))
                for line in open(class_file)
                if "_rlnAccuracyRotations" in line
            ][0] - 1
            if "2d" in args.parfile.lower():
                # keep 2D classes with rotation accuracy <= 1
                classes = numpy.array(
                    [
                        line.split()
                        for line in open(class_file)
                        if "classes.mrcs" in line
                    ]
                )
            else:
                # keep 3D classes with rotation accuracy <= 1.5
                classes = numpy.array(
                    [line.split() for line in open(class_file) if "_class0" in line]
                )

            prs = numpy.array(classes[:, rlnAccuracyRotations], dtype="f")
            from sklearn import mixture

            gmix = mixture.GMM(n_components=2, covariance_type="full")
            # gmix.fit( prs[ prs < 999 ].ravel())
            # cutoff = numpy.array( gmix.means_ ).mean()
            cutoff = 2

            # auto class selection based on model star file
            if not "auto" in args.classes.lower():
                # use base-1 indexes for compatibility with relion
                selected_classes = numpy.array(args.classes.split(","), dtype="i") - 1
            else:
                logger.info(
                    "Bimodal rlnAccuracyRotations distribution threshold %0.8f", cutoff
                )
                selected_classes = numpy.squeeze(
                    numpy.argwhere(
                        numpy.array(classes[:, rlnAccuracyRotations], dtype="f")
                        <= cutoff
                    ),
                    axis=(1,),
                )

            logger.info("Keeping classes %s", selected_classes + 1)

            # select multiple classes
            selected = []
            for selected_class in selected_classes:
                current = input[
                    input[:, rlnClassNumber] == str(int(selected_class + 1))
                ]
                if selected == []:
                    selected = current
                else:
                    selected = numpy.append(selected, current, axis=0)

            # create stack with selected classes
            if "2d" in args.parfile.lower():
                command = "{2}/bin/newstack {0}.mrcs {0}_selected.mrcs -secs {1}".format(
                    args.parfile.replace("data.star", "classes"),
                    ",".join([str(s) for s in selected_classes]),
                    get_imod_path(),
                )
                run_shell_command(command)

            global_indexes_to_keep = [
                int(current.split("@")[0]) - 1 for current in selected[:, rlnImageName]
            ]
            global_indexes_to_remove = []
            [
                global_indexes_to_remove.append(i)
                for i in range(input.shape[0])
                if i not in global_indexes_to_keep
            ]

            logger.info(
                "Selecting %i out of %i classes", len(selected_classes), prs.size
            )

        else:

            global_indexes_to_keep = sorted(
                [int(current.split("@")[0]) - 1 for current in input[:, rlnImageName]]
            )
            global_indexes_to_remove = []

            # get total number of particles from stack
            stack = input[0, rlnImageName].split("@")[1]
            # revert to original stack name
            if "/lscratch" in stack:
                stack = os.getcwd() + "/relion/" + stack[18:]

            total_number_of_particles = int(mrc.readHeaderFromFile(stack)["nz"])
            [
                global_indexes_to_remove.append(i)
                for i in range(total_number_of_particles)
                if i not in global_indexes_to_keep
            ]

            selected = input

        logger.info("Particles to remove = %i", len(global_indexes_to_remove))
        logger.info("Particles to keep = %i", len(global_indexes_to_keep))

        # indexes are in base-0
        if not args.debug:

            parameters = project_params.load_pyp_parameters()
            pixel_size = (
                float(parameters["scope_pixel"])
                * float(parameters["data_bin"])
                * float(parameters["extract_bin"])
            )

            # produce_boxx_files( global_indexes_to_remove, args.extract_cls )
            if args.shifts:
                shifts = -pixel_size * selected[:, [rlnOriginX, rlnOriginY]].astype("f")
                produce_boxx_files_fast(
                    global_indexes_to_remove, args.extract_cls, -shifts
                )
            else:
                produce_boxx_files_fast(global_indexes_to_remove, args.extract_cls)

            # create clean frealign parameter file
            if class_file != args.parfile and "/Refine3D" in args.parfile:

                # create new star file with rotation parameters

                # produce frealign parameter file
                outputparfile = args.parfile.replace(".star", "_clean_scores.par")
                f = open(outputparfile, "wb")
                if not args.phases:
                    f.writelines(frealign_parfile.NEW_PAR_HEADER)
                else:
                    f.writelines(frealign_parfile.CCLIN_PAR_HEADER)

                for i in range(len(global_indexes_to_keep)):
                    if not args.phases:
                        #           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC     -LogP      SIGMA   SCORE  CHANGE
                        # occ = 100, sigma = .5, logp = score = change = 0
                        f.write(
                            frealign_parfile.NEW_PAR_STRING_TEMPLATE
                            % (
                                i + 1,
                                float(selected[i, rlnAnglePsi]),
                                float(selected[i, rlnAngleTilt]),
                                float(selected[i, rlnAngleRot]),
                                -pixel_size * float(selected[i, rlnOriginX]),
                                -pixel_size * float(selected[i, rlnOriginY]),
                                float(parameters["scope_mag"]),
                                float(selected[i, rlnGroupNumber]) - 1,
                                float(selected[i, rlnDefocusU]),
                                float(selected[i, rlnDefocusV]),
                                float(selected[i, rlnDefocusAngle]),
                                100,
                                0,
                                0.5,
                                0,
                                0,
                            )
                        )
                    else:
                        f.write(
                            frealign_parfile.CCLIN_PAR_STRING_TEMPLATE
                            % (
                                i + 1,
                                float(selected[i, rlnAnglePsi]),
                                float(selected[i, rlnAngleTilt]),
                                float(selected[i, rlnAngleRot]),
                                -float(selected[i, rlnOriginX]),
                                -float(selected[i, rlnOriginY]),
                                float(parameters["scope_mag"]),
                                float(selected[i, rlnGroupNumber]) - 1,
                                float(selected[i, rlnDefocusU]),
                                float(selected[i, rlnDefocusV]),
                                float(selected[i, rlnDefocusAngle]),
                                0.0,
                                0.0,
                            )
                        )
                    f.write("\n")

                f.close()
                logger.info(
                    "FREALIGN parameter file for class(es) %s saved to %s",
                    args.classes,
                    outputparfile,
                )

                input = numpy.array(
                    [
                        line.split()
                        for line in open(outputparfile)
                        if not line.startswith("C")
                    ],
                    dtype=float,
                )

            if class_file != args.parfile:
                # create clean relion parameter file
                star_file = args.parfile
                clean_star_file = star_file.replace(".star", "_clean.star")
                com = "cat '{0}' | grep -v @ > '{1}'".format(star_file, clean_star_file)
                run_shell_command(com)

                sorted_input = input[numpy.argsort(input[:, rlnImageName])]

                with open(clean_star_file, "a") as arch:
                    for i in sorted(global_indexes_to_keep):
                        arch.write("\t".join(input[i, :].tolist()) + "\n")

                logger.info(
                    "RELION parameter file for class(es) %s saved to %s",
                    args.classes,
                    clean_star_file,
                )

        # for cls in selected_classes:
        #    com = """cat %s | grep @ | awk '{ if ($%s==%s) print }' >> %s""" % ( star_file, rlnClassNumber + 1, cls+1, clean_star_file )
        #    commands.getoutput( com )

    else:

        logger.error("Mode not recognized")



