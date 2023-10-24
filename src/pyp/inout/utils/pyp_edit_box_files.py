#!/usr/bin/env python

import multiprocessing
import os
import subprocess
import sys

import numpy

from pyp.system import project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def coordinates_from_mod_file(filename):

    modfile = subprocess.getoutput(
        "{0}/bin/imodinfo -a {1}".format(get_imod_path(), filename)
    ).split("contour")

    default_radius = (
        subprocess.getoutput(
            "{0}/bin/imodinfo -a {1}".format(get_imod_path(), filename)
        )
        .split("pointsize")[1]
        .split("\n")[0]
    )

    indexes = []

    if len(modfile) > 1:
        for c in range(1, len(modfile)):
            points_in_contour = int(modfile[c].split()[2])
            for point in range(points_in_contour):

                coord = modfile[c].split("\n")[point + 1].split()

                if ".vir" in filename and len(coord) == 3:
                    coord.append(default_radius)

                indexes.append(numpy.array(coord, dtype=float))
    return numpy.absolute(numpy.array(indexes))


# HF vir:
######################################################
def dimension_from_mod_file(filename):
    dimension = (
        subprocess.getoutput(
            "{0}/bin/imodinfo -a {1}".format(get_imod_path(), filename)
        )
        .split("max")[-1]
        .split("offsets")[0]
    )
    x, y, z = list(map(int, dimension.split()))
    return x, y, z


def indexes_from_mod_file(filename):

    # read coordinates from model file
    data = coordinates_from_mod_file(filename)

    indexes = []
    if data.size > 0:
        # get unique sorted list of z-coordinates
        indexes.extend(data[:, 2].squeeze().ravel().tolist())
        indexes = list(set(indexes))
        indexes.sort()
    return indexes


def indexes_from_class_file(filename):

    f = open(filename)
    A = numpy.array(f.read().split(), dtype=float)
    f.close()

    indexes = []

    pointer = 0
    while pointer < A.shape[0]:
        count = int(A[pointer + 1])
        indexes.append(
            (A[pointer + 3 : pointer + 3 + count] - 1).ravel().tolist()
        )  # indexes in IMAGIC start from 1
        pointer = pointer + 3 + count

    return indexes


def particle_selection_by_imagic_classification(modfile, lstfile):

    # read model file containing indexes of bad classes
    good_classes = indexes_from_mod_file(modfile)

    # read IMAGIC classification file to extract class membership information
    particle_indexes = indexes_from_class_file(lstfile)

    # build list of bad particles indexes
    global_indexes_to_keep = []
    for i in good_classes:
        global_indexes_to_keep.extend(particle_indexes[int(i)])

    return global_indexes_to_keep


def produce_boxx_files(modfile, lstfile):

    parameters = project_params.load_pyp_parameters()

    # read model file containing indexes of bad classes
    good_classes = indexes_from_mod_file(modfile)

    # read IMAGIC classification file to extract class membership information
    particle_indexes = indexes_from_class_file(lstfile)

    # build list of bad particles indexes
    global_indexes_to_keep = []
    for i in good_classes:
        global_indexes_to_keep.extend(particle_indexes[int(i)])
    global_indexes_to_keep.sort()


def produce_boxx_files(global_indexes_to_remove, classification_pass=1, shifts=[]):

    parameters = project_params.load_pyp_parameters()

    pixel = (
        float(parameters["scope_pixel"])
        * float(parameters["data_bin"])
        * float(parameters["extract_bin"])
    )

    micrographs = "{}.micrographs".format(parameters["data_set"])
    inputlist = [line.strip() for line in open(micrographs, "rb")]

    current_global_counter = previous_global_counter = micrograph_counter = 0

    boxx = numpy.array([1])

    global_indexes_to_remove.sort()

    # set last field in .boxx files to 0 for all bad particles
    for i in global_indexes_to_remove:

        while current_global_counter <= i:

            try:
                name = inputlist[micrograph_counter]
                boxxfile = "box/{}.boxx".format(name)
                boxfile = "box/{}.box".format(name)
            except:
                logger.exception(
                    "%d outside bounds %d", micrograph_counter, len(inputlist)
                )
                logger.exception("%d %d", current_global_counter, i)
                sys.exit()

            if os.path.exists(boxxfile):
                boxx = numpy.loadtxt(boxxfile, ndmin=2)
                box = numpy.loadtxt(boxfile, ndmin=2)

                if boxx.size > 0:
                    # only count particles that survived previous pass
                    valid_particles = boxx[:, 4].sum()
                    valid_particles = numpy.where(
                        numpy.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        1,
                        0,
                    ).sum()

                    previous_global_counter = current_global_counter
                    current_global_counter = (
                        current_global_counter + valid_particles
                    )  # 5th column contains actually extracted particles

                    # increment class membership pass for all particles of previous classification pass
                    boxxx = numpy.where(
                        numpy.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        classification_pass,
                        boxx[:, 5],
                    )

                    # alignments
                    if len(shifts) > 0:
                        # boxx[:,0:1] = box[:0:1] + numpy.round( input[ previous_global_counter : current_global_counter, 4:5 ] / pixel )
                        boxx[:, 0:1] = box[:0:1] + numpy.round(
                            shifts[previous_global_counter:current_global_counter, :]
                            / pixel
                        )

                    # save new extended box file
                    boxx[:, 5] = boxxx

                    numpy.savetxt(
                        "box/{}.boxx".format(name), boxx, fmt="%2.2f", delimiter="\t"
                    )
            else:
                logger.info("%s does not exist", boxxfile)

            micrograph_counter += 1

        index_in_micrograph = 0
        global_counter = -1

        # print i, previous_global_counter
        while global_counter < i - previous_global_counter:
            if (
                boxx[index_in_micrograph, 4] == 1
                and boxx[index_in_micrograph, 5] >= classification_pass - 1
            ):
                global_counter += 1
            index_in_micrograph += 1
        # boxx[index-1,5] = 0
        boxxx[index_in_micrograph - 1] = classification_pass - 1

        boxx[:, 5] = boxxx

        """
        # adjust particle shifts
        if alignment_parameters:
            newx = float(alignment_parameters[i,4]) / pixel
            newy = float(alignment_parameters[i,5]) / pixel
            # correct particle positions
            boxx[index_in_micrograph-1,0] += numpy.round( newx )
            boxx[index_in_micrograph-1,1] += numpy.round( newy )
            # update image shifts
            alignment_parameters[i,4] = newx - numpy.round( newx )
            alignment_parameters[i,5] = newy - numpy.round( newy )
        """

        # save new extended box file
        numpy.savetxt("box/{}.boxx".format(name), boxx, fmt="%2.2f", delimiter="\t")


def read_boxx_file_async(micrograph, results):
    boxxfile = "box/{}.boxx".format(micrograph)
    boxx = numpy.loadtxt(boxxfile, ndmin=2)
    boxfile = "box/{}.box".format(micrograph)
    box = numpy.loadtxt(boxfile, ndmin=2)
    results.put([micrograph, boxx, box])


def write_boxx_file_async(micrograph, boxx):
    numpy.savetxt("box/{}.boxx".format(micrograph), boxx, fmt="%2.2f", delimiter="\t")


def produce_boxx_files_fast(global_indexes_to_remove, classification_pass=1, shifts=[]):

    parameters = project_params.load_pyp_parameters()
    pixel = float(parameters["scope_pixel"]) * float(parameters["data_bin"])

    micrographs = "{}.micrographs".format(parameters["data_set"])
    inputlist = [line.strip() for line in open(micrographs, "r")]

    current_global_counter = previous_global_counter = micrograph_counter = 0

    boxx = numpy.array([1])

    global_indexes_to_remove.sort()

    threads = 12

    # read all boxx files in parallel
    pool = multiprocessing.Pool(threads)
    manager = multiprocessing.Manager()
    results = manager.Queue()
    logger.info("Reading box files using %i threads", threads)
    for micrograph in inputlist:
        pool.apply_async(read_boxx_file_async, args=(micrograph, results))
        # read_boxx_file_async( micrograph, results )
        # boxxfile = 'box/{}.boxx'.format(micrograph)
        # master_list[ micrograph_counter ] = numpy.loadtxt( boxxfile, ndmin=2 )
    pool.close()

    # Wait for all processes to complete
    pool.join()

    boxx_dbase = dict()
    box_dbase = dict()
    while results.empty() == False:
        current = results.get()
        boxx_dbase[current[0]] = current[1]
        box_dbase[current[0]] = current[2]

    local_counter = 0

    # add infinity to force processing of micrograph list to the end
    import sys

    # global_indexes_to_remove.append( sys.maxint )

    logger.info("Updating particle database")

    # set last field in .boxx files to 0 for all bad particles
    for i in global_indexes_to_remove:

        while current_global_counter <= i:

            try:
                name = inputlist[micrograph_counter]
                boxxfile = "box/{}.boxx".format(name)
                boxfile = "box/{}.box".format(name)
            except:
                logger.exception(
                    "%d outside bounds %d", micrograph_counter, len(inputlist)
                )
                logger.exception("%d %d", current_global_counter, i)
                sys.exit()

            if os.path.exists(boxxfile):
                # boxx = numpy.loadtxt( boxxfile, ndmin=2 )
                boxx = boxx_dbase[name]
                box = box_dbase[name]

                if boxx.size > 0:

                    # only count particles that survived previous pass
                    # valid_particles = boxx[:,4].sum()
                    valid_particles = numpy.where(
                        numpy.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        1,
                        0,
                    ).sum()

                    previous_global_counter = current_global_counter
                    current_global_counter = (
                        current_global_counter + valid_particles
                    )  # 5th column contains actually extracted particles

                    # increment class membership pass for all particles of previous classification pass
                    boxxx = numpy.where(
                        numpy.logical_and(
                            boxx[:, 4] > 0, boxx[:, 5] >= classification_pass - 1
                        ),
                        classification_pass,
                        boxx[:, 5],
                    )

                    # alignments
                    counter = previous_global_counter
                    for j in range(boxx.shape[0]):
                        if boxx[j, 4] > 0 and boxx[j, 5] >= classification_pass - 1:
                            # print box[j,0:2]
                            # print numpy.round( input[ counter, 4:6 ] / pixel )
                            if len(shifts) > 0:
                                # boxx_dbase[name][j,0:2] = box[j,0:2] - boxx[j,2:4] / 2 - numpy.round( input[ counter, 4:6 ] / pixel )
                                boxx_dbase[name][j, 0:2] = (
                                    box[j, 0:2]
                                    - boxx[j, 2:4] / 2
                                    + numpy.round(shifts[local_counter, :] / pixel)
                                )
                                # boxx_dbase[name][j,0] = box[j,0] - boxx[j,2] / 2 + numpy.round( shifts[ local_counter, 1 ] / pixel )
                                # boxx_dbase[name][j,1] = box[j,1] - boxx[j,3] / 2 + numpy.round( shifts[ local_counter, 0 ] / pixel )
                                local_counter += 1
                            else:
                                boxx_dbase[name][j, 0:2] = (
                                    box[j, 0:2] - boxx[j, 2:4] / 2
                                )
                            # print boxx[j,0:2]
                            # sys.exit()
                            counter += 1

                    # save new extended box file
                    # boxx[:,5] = boxxx
                    boxx_dbase[name][:, 5] = boxxx
                    # numpy.savetxt('box/{}.boxx'.format(name),boxx,fmt='%2.2f',delimiter='\t')
            else:
                logger.info("%s does not exist", boxxfile)

            if micrograph_counter < len(inputlist) - 1:
                micrograph_counter += 1
            # if micrograph_counter == len(inputlist):
            else:
                logger.info("Reached the end")
                break

        if micrograph_counter < len(inputlist):

            index_in_micrograph = 0
            global_counter = -1

            # print i, previous_global_counter
            while global_counter < i - previous_global_counter:
                if (
                    boxx[index_in_micrograph, 4] == 1
                    and boxx[index_in_micrograph, 5] >= classification_pass - 1
                ):
                    global_counter += 1
                index_in_micrograph += 1
            # boxx[index-1,5] = 0
            boxxx[index_in_micrograph - 1] = classification_pass - 1
            # print boxxx[index_in_micrograph-1]
            # sys.exit()

            # boxx[:,5] = boxxx
            boxx_dbase[name][:, 5] = boxxx
            # print 'Now parsing', name

            # save new extended box file
            # numpy.savetxt('box/{}.boxx'.format(name),boxx,fmt='%2.2f',delimiter='\t')

    logger.info("Current global count %d", current_global_counter)
    # save all boxx files in parallel
    pool = multiprocessing.Pool(threads)
    manager = multiprocessing.Manager()
    results = manager.Queue()
    logger.info("Saving box files using %i threads", threads)
    for micrograph in inputlist:
        pool.apply_async(
            write_boxx_file_async, args=(micrograph, boxx_dbase[micrograph])
        )
        # write_boxx_file_async( micrograph, boxx_dbase[micrograph] )
    pool.close()

    # Wait for all processes to complete
    pool.join()
    return boxx_dbase

def produce_box_files(modfilename, listfilename, binning=8, size=256):

    # read list of images
    images = [
        os.path.splitext(os.path.basename(s.split()[1]))[0]
        for s in open(listfilename)
        if s.startswith("IMAGE")
    ]

    # read coordinates from model file
    modfile = subprocess.getoutput(
        "{0}/bin/imodinfo -a {1}".format(get_imod_path(), modfilename)
    )

    # deduct image size
    sizey = int(modfile[modfile.find("max") :].split()[2]) * binning - binning / 2

    # extract coordinates
    mod = modfile.split("contour")
    for c in range(1, len(mod)):
        count = int(mod[c].split()[2])
        if c == 1:
            data = numpy.array(
                mod[c].split()[3 : 3 + 3 * count], dtype="float"
            ).reshape([count, 3])
        else:
            data = numpy.concatenate(
                (
                    data,
                    numpy.array(
                        mod[c].split()[3 : 3 + 3 * count], dtype="float"
                    ).reshape([count, 3]),
                ),
                axis=0,
            )

    ## remove .box files if existing
    # for f in glob.glob('box/*.box'):
    #    os.remove(f)

    # save box files
    counter = 0
    for i in range(len(images)):

        current = data[data[:, 2] == i]

        if current.size > 0:
            # save new extended box file
            box = numpy.column_stack(
                (
                    current[:, 0] * binning - size / 2,
                    sizey - current[:, 1] * binning - size / 2,
                    size * numpy.ones(current.shape[0]),
                    size * numpy.ones(current.shape[0]),
                )
            )
            numpy.savetxt(
                "box/{}.box".format(images[i]), box, fmt="%2.0f", delimiter="\t"
            )
            counter += 1

    logger.info(
        "{0} particles extracted from {1} micrographs.".format(data.shape[0], counter)
    )


def extract_irregular_regions(modfile):

    coordinates = []

    CutCoordTxt = modfile.replace(".mod", ".txt")
    command = (
        get_imod_path()
        + "/bin/model2point -ObjectAndContour -input "
        + modfile
        + " -output "
        + CutCoordTxt
    )
    local_run.run_shell_command(command)

    CutCoord_Fl = open(CutCoordTxt, "rb")
    CutCoord = CutCoord_Fl.readlines()
    CutCoord_Fl.close()

    os.remove(CutCoordTxt)

    if len(CutCoord) < 3:
        return 1  #  there are boxing sign in the file. stop here

    Obj_old = -1
    for (q1, CutCoord_q1) in enumerate(CutCoord):
        Obj_Cntr_xyz_q1 = CutCoord_q1.split()
        Obj_q1 = int(Obj_Cntr_xyz_q1[0])
        # Cntr_q1= int(Obj_Cntr_xyz_q1[1])  # The program is indiferrent ot the countur number
        X_q1 = int(Obj_Cntr_xyz_q1[2])
        Y_q1 = int(Obj_Cntr_xyz_q1[3])
        Z_q1 = int(Obj_Cntr_xyz_q1[4])

        # intiating the coordinates of the 1st box (this happens only one when the old object number is -1 as was set before the loop)
        if Obj_old == -1:
            CurrentObj_X = [X_q1]
            CurrentObj_Y = [Y_q1]
            CurrentObj_Z = [Z_q1]
        # extending the coordinates of the curent object
        elif Obj_old == Obj_q1:
            CurrentObj_X.append(X_q1)
            CurrentObj_Y.append(Y_q1)
            CurrentObj_Z.append(Z_q1)
        # A new object was detected - process the preceding object and initiate the new object
        else:

            max_X = max(CurrentObj_X)
            max_Y = max(CurrentObj_Y)
            max_Z = max(CurrentObj_Z)

            min_X = min(CurrentObj_X)
            min_Y = min(CurrentObj_Y)
            min_Z = min(CurrentObj_Z)

            delta_X = max_X - min_X
            delta_Y = max_Y - min_Y
            delta_Z = max_Z - min_Z

            center_X = min_X + delta_X / 2
            center_Y = min_Y + delta_Y / 2
            center_Z = min_Z + delta_Z / 2

            coordinates.append(
                [center_X, center_Y, center_Z, delta_X, delta_Y, delta_Z]
            )

            # initiate the new object
            CurrentObj_X = [X_q1]
            CurrentObj_Y = [Y_q1]
            CurrentObj_Z = [Z_q1]

        # * end if
        # Updating the object-number before reading the next line
        Obj_old = Obj_q1

    max_X = max(CurrentObj_X)
    max_Y = max(CurrentObj_Y)
    max_Z = max(CurrentObj_Z)

    min_X = min(CurrentObj_X)
    min_Y = min(CurrentObj_Y)
    min_Z = min(CurrentObj_Z)

    delta_X = max_X - min_X
    delta_Y = max_Y - min_Y
    delta_Z = max_Z - min_Z

    center_X = min_X + delta_X / 2
    center_Y = min_Y + delta_Y / 2
    center_Z = min_Z + delta_Z / 2

    coordinates.append([center_X, center_Y, center_Z, delta_X, delta_Y, delta_Z])

    return coordinates


def regular_points_from_line(modfile, spacing):

    contours = subprocess.getoutput(
        "{0}/bin/imodinfo -a {1}".format(get_imod_path(), modfile)
    ).split("contour")
    points = []
    basename = os.path.splitext(modfile)[0]

    if len(contours) > 1:
        for c in range(1, len(contours)):
            line_start = numpy.array(contours[c].split("\n")[1].split(), dtype=float)
            line_end = numpy.array(contours[c].split("\n")[2].split(), dtype=float)
            line_norm = numpy.linalg.norm(line_end - line_start)
            step = spacing * (line_end - line_start) / line_norm
            delta = numpy.linalg.norm(step)

            for sec in range(0, int(line_norm // delta) + 1):
                points.append(line_start + sec * step)
                points.append(line_end)

        b_array = numpy.array(points, dtype=int)
        boxarray = numpy.round(b_array, 1)
        # mod file read with imodinfo is X, Y, Z
        read_mod = "{}.xyz".format(basename)
        # box file for new mod file generation should be X,Z,Y
        boxfile = "{}.box".format(basename)
        numpy.savetxt(read_mod, boxarray, fmt="%.1f", delimiter="\t")

        # swap yz in box file
        command = "awk '{{print $1,$3,$2}}' {0} > {1}".format(read_mod, boxfile)
        local_run.run_shell_command(command)

        command = "{0}/bin/point2model -scat -sphere {1} {2}.box {2}.mod".format(
            get_imod_path(), int(spacing) * 2, basename
        )
        local_run.run_shell_command(command)
    return numpy.array(points)


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

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Particle selection based on results of classification."
    )
    # parser.add_argument("-mode", help='Mode of operation: mod2box (imod model file to box), box2mod (box to imod model file), frealign2box (frealign parameters file to box), relion2box (relion star file to box), or imagic2box (IMAGIC classification to box), eman2box (EMAN classification to box)', required=True)
    # parser.add_argument("-modfile", help='IMOD model file with selected BAD classes')
    # parser.add_argument("-listfile", help='Imagic classification file')
    parser.add_argument(
        "-parfile", help="File with GOOD particles produce by e2evalparticles.py"
    )
    parser.add_argument("-extract_cls", help="Classification pass.", type=int)
    # parser.add_argument("-particles", help='Total number of particles before classification.', type=int)
    # parser.add_argument("-binning", help='Binning factor used for picking', type=int, default=8)
    # parser.add_argument("-boxsize", help='Default box size for particles', type=int, default=256)
    parser.add_argument(
        "-threshold",
        help="Threshold value for scores used for particle selection.",
        type=str,
        default="0",
    )
    parser.add_argument(
        "-occupancy",
        help="Threshold value for occupancy used for particle selection.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-classes",
        help="Classes to keep (0-based indexes separated by ,)",
        type=str,
        default="",
    )
    parser.add_argument(
        "-shifts",
        help="Reset particle shifts (FREALIGN ONLY)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-phases",
        help="Parameters contain phase residuals instead of scores (FREALIGN ONLY)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-debug", help="Do not write .boxx files.", action="store_true", default=False
    )
    args = parser.parse_args()

    parfile_extension = os.path.splitext(args.parfile)[-1]
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
            local_run.run_shell_command(command)
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

            logger.info("Particles to remove = %i", len(global_indexes_to_remove))
            logger.info("Particles to keep = %i", len(global_indexes_to_keep))

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
            par_files = glob.glob(args.parfile)

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
            logger.info("global_indexes_to_remove %s", global_indexes_to_remove[:5])
            logger.info("global_indexes_to_keep %s", global_indexes_to_keep[:5])

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

            logger.info("Particles to remove = %i", len(global_indexes_to_remove))
            logger.info("Particles to keep = %i", len(global_indexes_to_keep))

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

                outputparfile = args.parfile.replace(".par", "_clean.par").replace(
                    "?", "1"
                )
                f = open(outputparfile, "w")
                for line in open(args.parfile.replace("?", "1")):
                    if line.startswith("C"):
                        f.write(line)

                if args.shifts:
                    parameters = project_params.load_pyp_parameters()
                    pixel = float(parameters["scope_pixel"]) * float(
                        parameters["data_bin"]
                    )
                    newinput[:, 4:6] -= numpy.round(newinput[:, 4:6] / pixel) * pixel

                for i in range(newinput.shape[0]):
                    if not args.phases:
                        f.write(
                            frealign_parfile.NEW_PAR_STRING_TEMPLATE
                            % tuple(newinput[i, :17])
                        )
                    else:
                        f.write(
                            frealign_parfile.CCLIN_PAR_STRING_TEMPLATE
                            % tuple(newinput[i, :13])
                        )
                    f.write("\n")
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
                local_run.run_shell_command(command)

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
                com = "cat {0} | grep -v @ > {1}".format(star_file, clean_star_file)
                local_run.run_shell_command(com)

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
