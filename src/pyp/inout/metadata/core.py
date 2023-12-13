import collections
import csv
import glob
import math
from operator import index
import os
import re
import shutil
import sys
import time
import json
import random
from math import cos, sin
from pathlib import Path, PosixPath
from xml.dom.minidom import parse

import numpy as np

from pyp.analysis.geometry import (
    DefocusOffsetFromCenter,
    getShiftsForRecenter,
    spa_euler_angles,
)
from pyp.inout.image import mrc
from pyp.inout.metadata import frealign_parfile, pyp_metadata
from pyp.inout.utils import pyp_edit_box_files as imod
from pyp.system import local_run, project_params
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path
from pyp.utils import get_relative_path, movie2regex
from pyp.utils import timer, symlink_relative
from pyp.analysis.geometry import get_vir_binning_boxsize

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

logger = initialize_pyp_logger()


def isfrealignx(parfile):
    with open(parfile) as f:
        # input = [line for line in f.read().split("\n") if not line.startswith("C")]
        input = frealign_parfile.Parameters.from_file(parfile).data

        if len(input[0]) == 46 or len(input[0]) == 17:
            is_frealignx = True
        else:
            is_frealignx = False
    return is_frealignx


# generate RELION parameter file
def generateRelionParFileNew(
    inputlist, name, data_bin, cs, wgh, parameters, astigmatism="False"
):

    stacklist = [
        "relion/Particles/Micrographs/" + line + "_particles.mrcs" for line in inputlist
    ]
    defocuslist = ["ctf/" + line.strip() + ".ctf" for line in inputlist]
    boxlist = ["box/" + line.strip() + ".boxx" for line in inputlist]

    # if not 'frames' in parameters['extract_fmt']:
    if True:
        header = """data_\nloop_\n_rlnMicrographName #1\n_rlnImageName #2\n_rlnDefocusU #3\n_rlnDefocusV #4\n_rlnDefocusAngle #5\n_rlnVoltage #6\n_rlnSphericalAberration #7\n_rlnAmplitudeContrast #8\n_rlnMagnification #9\n_rlnDetectorPixelSize #10\n_rlnCtfFigureOfMerit #11\n_rlnGroupName #12\n_rlnGroupNumber #13\n"""
        f = open("relion/{0}.star".format(name), "w")
        f.write(header)
        fp = open("relion/{0}_particles.star".format(name), "w")
        fp.write(header)
        count = 1
        film = 0

        defocuses = []
        total_number_of_particles = 0
        for defocus, boxxs in zip(defocuslist, boxlist):
            ctf = np.loadtxt(defocus)
            defocuses.append((float(ctf[2]) + float(ctf[3])) / 2.0)
            boxx = np.loadtxt(boxxs, ndmin=2)
            total_number_of_particles += boxx[
                np.logical_and(
                    boxx[:, 4] == 1, boxx[:, 5] >= int(parameters["extract_cls"])
                )
            ].shape[0]
        defocus = np.array(defocuses, dtype="f")
        defocus_min_value = defocus.min()
        # use ~50 particles per defocus group
        defocus_groups = int(total_number_of_particles / 1000.0)
        defocus_group_range = (defocus.max() - defocus_min_value) / defocus_groups

        for sname, stack, defocus, boxxs in zip(
            inputlist, stacklist, defocuslist, boxlist
        ):

            pcount = 1
            if os.path.exists(stack) or os.path.exists(boxxs):
                ctf = np.loadtxt(defocus)
                # print ctf.shape[0]
                if ctf.shape[0] < 12:
                    logger.error(
                        "Not enough parameters in %s. Re-run ctf estimation.", defocus
                    )
                    return
                if os.path.exists(stack):
                    number_of_particles = mrc.readHeaderFromFile(stack)["nz"]
                else:
                    boxx = np.loadtxt(boxxs)
                    # number_of_particles = boxx[ boxx[:,-2] + boxx[:,-1] == 2 ].shape[0]
                    number_of_particles = boxx[
                        np.logical_and(
                            boxx[:, 4] == 1,
                            boxx[:, 5] >= int(parameters["extract_cls"]),
                        )
                    ].shape[0]
                    # print number_of_particles

                if number_of_particles == 0:
                    logger.info("No particles found for {}".format(sname))
                    inputlist.remove(sname)
                else:
                    # print sname, number_of_particles
                    for p in range(number_of_particles):
                        if "F" in astigmatism:
                            df1 = df2 = ctf[0]  # TOMOCTFFIND
                            angast = 45.0
                        else:
                            df1 = ctf[2]  # CTFFIND3
                            df2 = ctf[3]  # CTFFIND3
                            angast = ctf[4]  # CTFFIND3
                        ccc = ctf[5]  # CTFFIND3
                        pixel = ctf[9] * float(data_bin)
                        voltage = ctf[10]
                        magnification = ctf[11] / float(data_bin)
                        dstep = float(pixel) * magnification / 10000.0
                        defocus_group = (
                            int(
                                ((float(df1) + float(df2)) / 2.0 - defocus_min_value)
                                / defocus_group_range
                            )
                            + 1
                        )
                        defocus_group_name = "group_%03d" % defocus_group
                        f.write(
                            "%s %.6i@%s/relion/%s_stack.mrcs %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %s %d\n"
                            % (
                                sname,
                                count,
                                os.getcwd(),
                                name,
                                df1,
                                df2,
                                angast,
                                float(voltage),
                                float(cs),
                                float(wgh),
                                magnification,
                                dstep,
                                ccc,
                                defocus_group_name,
                                defocus_group,
                            )
                        )
                        fp.write(
                            "%s %.6i@Particles/Micrographs/%s_particles.mrcs %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %s %d\n"
                            % (
                                sname,
                                pcount,
                                sname,
                                df1,
                                df2,
                                angast,
                                float(voltage),
                                float(cs),
                                float(wgh),
                                magnification,
                                dstep,
                                ccc,
                                defocus_group_name,
                                defocus_group,
                            )
                        )
                        count += 1
                        pcount += 1
                    film += 1
            else:
                logger.info("No particles found for {}".format(sname))
                inputlist.remove(sname)
        f.close()
        fp.close()

    # merge movie files
    if "relion_frames" in parameters["extract_fmt"].lower():
        f = open("relion/{0}_frames.star".format(name), "w")
        star_file_list = [
            "relion/" + line.strip() + "_frames.star" for line in inputlist
        ]
        # write header
        f.write(
            """\ndata_\n\nloop_\n_rlnOriginalParticleName #1\n_rlnCoordinateX #2\n_rlnCoordinateY #3\n_rlnImageName #4\n_rlnDefocusU #5\n_rlnDefocusV #6\n_rlnDefocusAngle #7\n_rlnVoltage #8\n_rlnSphericalAberration #9\n_rlnAmplitudeContrast #10\n_rlnMagnification #11\n_rlnDetectorPixelSize #12\n_rlnCtfFigureOfMerit #13\n_rlnAnglePsi #14\n_rlnAutopickFigureOfMerit #15\n_rlnClassNumber #16\n_rlnMicrographName #17\n"""
        )

        """
        _rlnMicrographName #1
        _rlnCoordinateX #2
        _rlnCoordinateY #3
        _rlnImageName #4
        _rlnDefocusU #5
        _rlnDefocusV #6
        _rlnDefocusAngle #7
        _rlnVoltage #8
        _rlnSphericalAberration #9
        _rlnAmplitudeContrast #10
        _rlnMagnification #11
        _rlnDetectorPixelSize #12
        _rlnCtfFigureOfMerit #13
        _rlnAnglePsi #14
        _rlnAutopickFigureOfMerit #15
        _rlnClassNumber #16
        _rlnOriginalParticleName #17
        _rlnGroupName #18
        _rlnGroupNumber #19
        """

        count = 0

        # merge all star files
        for stacks, particles, boxes, series in zip(
            star_file_list, stacklist, boxlist, inputlist
        ):
            # restart counter because we are keeping one frame stack per micrograph
            count = 0
            logger.info("Processing %s", stacks)
            with open(stacks, "r") as infile:
                for line in infile:
                    # update film number
                    # f.write( '%.6i@%s/relion/%s_stack.mrcs\t' % ( count + int(line.split()[0]), os.getcwd(), name ) + '\t'.join(line.split()[1:]) + '\n' )
                    f.write(
                        "%.6i@Particles/Micrographs/%s_particles.mrcs\t"
                        % (count + int(line.split()[0]), series)
                        + "\t".join(line.split()[1:])
                        + "\n"
                    )
                    # f.write( '%.6i@%s/%s\t' % ( count + int(line.split()[0]), os.getcwd(), particles ) + '\t'.join(line.split()[1:]) + '\n' )
                    # f.write( '%.6i@%s/%s\t' % ( count + int(line.split()[0]), os.getcwd(), particles ) + '\t'.join(line.split()[1:]) + '\n' )
                    # count += 1
            # number_of_particles = mrc.readHeaderFromFile(particles)['nz']
            boxxs = np.loadtxt(boxes, ndmin=2)
            number_of_particles = boxxs[
                np.logical_and(
                    boxxs[:, 4] == 1, boxxs[:, 5] >= int(parameters["extract_cls"])
                )
            ].shape[0]
            # count += number_of_particles
        f.close()

        # remove individual star files
        [os.remove(s) for s in star_file_list if os.path.exists(s)]


def generateRelionTomogramPositions(name, parameters):

    cutboxsize = int(parameters["extract_box"])

    # set virion box size
    virion_boxsize = 3 * int(parameters["tomo_vir_rad"] / parameters["scope_pixel"])

    # Binning factor used for particle picking (coordinates in .vir file)
    bintomo_bin = 4.0

    # Binning factor of reconstructed particles (This is usually 1, but for smaller pixel sizes one may use binned data for 3DAVG to work properly)
    final_bin = 1.0

    binning = bintomo_bin * final_bin

    pixel = (
        float(parameters["scope_pixel"])
        * float(parameters["data_bin"])
        * float(parameters["extract_bin"])
    )

    sub_tomogram_positions = []

    for f in "mod/{0}.vir mrc/{0}.rec ctf/{0}.ctf".format(name).split():
        if not os.path.isfile(f):
            logger.warning("File %s not found." % f)
            return sub_tomogram_positions

    # get size of full unbinned reconstruction from .rec file
    rec_header = mrc.readHeaderFromFile("mrc/%s.rec" % name)
    recX, recY, recZ = [
        binning * x
        for x in [int(rec_header["nx"]), int(rec_header["nz"]), int(rec_header["ny"])]
    ]

    virion_coordinates = imod.coordinates_from_mod_file("mod/%s.vir" % name)

    # traverse all virions in tilt series
    for vir in range(virion_coordinates.shape[0]):

        vir_x, vir_y, vir_z = [
            binning * virion_coordinates[vir, 0],
            binning * virion_coordinates[vir, 1],
            recZ - binning * virion_coordinates[vir, 2],
        ]

        # check if we have picked spikes for this virion
        virion_file = "sva/%s_vir%04d_cut.txt" % (name, vir)
        if not os.path.isfile(virion_file):
            logger.warning("File %s not found. Skipping.", virion_file)
            continue
        else:
            spikes_in_virion = np.loadtxt(
                virion_file, comments="number", usecols=(list(range(32))), ndmin=2
            )
            if spikes_in_virion.shape[0] == 0:
                logger.warning("File %s not found. Skipping.", virion_file)
                continue

        # for all spikes in current virion
        for spike in range(spikes_in_virion.shape[0]):

            # extract local spike coordinates [0-479]
            spike_x, spike_y, spike_z = spikes_in_virion[spike, 3:6]

            # change y-coordinate to imod convention
            spike_y = virion_boxsize - 1 - spike_y

            # THIS IS TO MATCH OLDER EXPERIMENTS
            spike_y += 1

            logger.info(
                "Processing spike %d at x,y,z coordinates [ %.1f, %.1f, %.1f ]",
                spike,
                spike_x,
                spike_y,
                spike_z,
            )

            # compute global spike coordinates from virus box size
            spike_X = vir_x + (spike_x - virion_boxsize // 2) * final_bin + 1 * 0.5
            spike_Y = vir_y - (spike_y - virion_boxsize // 2) * final_bin + 0 * 0.5
            # spike_Y = vir_y + (spike_y-virion_boxsize//2) * final_bin + 0 *.5
            spike_Z = vir_z + (spike_z - virion_boxsize // 2) * final_bin + 1 * 0.5

            if False:
                logger.info(
                    "[spike_X, spike_Y, spike_Z ] = [%s, %s, %s]",
                    spike_X,
                    spike_Y,
                    spike_Z,
                )

            sub_tomogram_positions.append([spike_X, spike_Y, spike_Z])

    return sub_tomogram_positions


# compute average frame weights across dataset
def generateGlobalFrameWeights(dataset):

    # weightslist = [ 'ali/' + line.strip() + '_weights.txt' for line in inputlist ]

    weightslist = glob.glob("ali/*_weights.txt")

    average_weights = glob.glob("ali/{}_???_weights.txt".format(dataset))

    # ignore previous weight averages
    for average in average_weights:
        try:
            weightslist.remove(average)
        except:
            pass

    all_weights = dict()
    all_counters = dict()

    for weights in weightslist:

        current_weights = np.loadtxt(weights, ndmin=2).mean(axis=0)
        frames = current_weights.size
        if str(frames) not in list(all_weights.keys()):
            all_weights[str(frames)] = np.zeros(frames)
            all_counters[str(frames)] = 0
        all_weights[str(frames)] += current_weights
        all_counters[str(frames)] += 1

    for i in all_weights.keys():
        all_weights[i] /= all_counters[i]
        all_weights[i] /= all_weights[i].sum()

        logger.info("Saving weights to ali/%s_%03d_weights.txt" % (dataset, int(i)))

        np.savetxt("ali/%s_%03d_weights.txt" % (dataset, int(i)), all_weights[i])


def readMDOCfile(docfile):

    if os.path.isfile(docfile):
        d = {}
        with open(docfile) as f:
            for line in f:
                if len(line.split("=")) == 2:
                    (key, val) = line.split("=")
                    if not "DateTime" in key and "PixelSpacing" not in list(d.keys()):
                        d[key] = val

        pixel_size = float(d["PixelSpacing "])
        voltage = 300.0
        mag = float(d["Magnification "])
        defocus = float(d["Defocus "]) * 1e10

        return [pixel_size, voltage, mag, defocus]
    else:
        logger.info("File {0}/{1} does not exist.".format(os.getcwd(), docfile))
        return [0, 0, 0, 0]


def readMRCheader(file):

    if os.path.isfile(file):
        docfile = file.replace(".mrc", ".mrc.mdoc")

        if os.path.isfile(docfile):
            pixel_size, voltage, mag, defocus = readMDOCfile(docfile)
            tilt_axis = 0
        else:
            h = mrc.readHeaderFromFile(file)
            sizex = int(h["mx"])
            if sizex > 6096:
                binning = 1
            else:
                binning = 2
            pixel_size = sizex / float(h["nx"])
            pixel_size *= binning
            # pixel_size = float([line.split()[1] for line in header if 'apix_x' in line][0])
            voltage = 300.0
            mag = 10000.0
            defocus = 20000.0
            tilt_axis = 0

        return [pixel_size, voltage, mag, defocus, tilt_axis]
    else:
        logger.info("File {0} does not exist.".format(file))


def readXMLfile(xmlfile):

    if os.path.isfile(xmlfile):
        knownpaths = collections.OrderedDict(
            [
                ("Pixel Size", "pixelSize x numericValue"),
                (
                    "AccelerationVoltage",
                    "MicroscopeImage microscopeData gun AccelerationVoltage",
                ),
                (
                    "Set Magnification",
                    "MicroscopeImage microscopeData optics TemMagnification NominalMagnification",
                ),
                (
                    "Applied Defocus",
                    "MicroscopeImage CustomData a:KeyValueOfstringanyType a:Value",
                ),
            ]
        )

        values = []
        for entry in knownpaths:
            dom = parse(xmlfile)
            for key in knownpaths[entry].split():
                if key == "a:KeyValueOfstringanyType":
                    dom = dom.getElementsByTagName(key)[-2]
                else:
                    dom = dom.getElementsByTagName(key)[-1]
            values.append(dom.childNodes[0].nodeValue)

        pixel_size = float(values[0]) * 1e10
        voltage = float(values[1]) * 1e-3
        mag = float(values[2])
        try:
            defocus = float(values[3]) * 1e10
        except:
            defocus = 25000
            pass
        return [pixel_size, voltage, mag, defocus]
    else:
        logger.info("File {0}/{1} does not exist.".format(os.getcwd(), xmlfile))


def compileDatabase(inputlist, filename):
    # row = {'Micrograph': 'None', 'Date and time': 'N/A', 'QC': 'good'}
    row = collections.OrderedDict(
        [("Micrograph", "None"), ("Date and time", "N/A"), ("Ignore", 0.0)]
    )
    row.update(
        [
            ("Magnification", 0.0),
            ("Voltage", 0.0),
            ("Exposure time", 0.0),
            ("Pixel size", 0.0),
            ("Applied defocus", 0.0),
        ]
    )
    row.update(
        [
            ("DF", 0.0),
            ("CC", 0.0),
            ("CCCC", 0.0),
            ("Counts", 0.0),
            ("DF1", 0.0),
            ("DF2", 0.0),
            ("Angast", 0.0),
            ("CCC", 0.0),
            ("X", 0),
            ("Y", 0),
            ("Z", 0),
        ]
    )
    row.update([("DF1-DF2", 0.0), ("CC_High_Res", 0.0), ("Area from .1 to .2", 0.0)])
    row.update([("Particles", 0), ("Avg_Drift", 0), ("Max_Drift", 0)])

    f = open(filename, "w")
    csvwriter = csv.DictWriter(f, delimiter="\t", fieldnames=list(row.keys()))
    csvwriter.writerow(dict((fn, fn) for fn in list(row.keys())))
    for i in inputlist:
        row["Micrograph"] = i
        currentfile = glob.glob("raw/" + i + "*")[0]
        row["Date and time"] = time.ctime(os.path.getctime(currentfile))
        row["Applied defocus"] = 0.0
        ctffile = "ctf/" + i + ".ctf"
        if os.path.isfile(ctffile):
            ctf = np.loadtxt(ctffile)
            row["X"], row["Y"], row["Z"] = ctf[6], ctf[7], ctf[8]
            row["DF"], row["CC"], row["DF1"], row["DF2"], row["Angast"], row["CCC"] = (
                ctf[0],
                ctf[1],
                ctf[2],
                ctf[3],
                ctf[4],
                ctf[5],
            )
            row["DF1-DF2"], row["CC_High_Res"] = abs(ctf[2] - ctf[3]), 0.0
            row["Pixel size"], row["Voltage"], row["Magnification"] = (
                ctf[9],
                ctf[10],
                ctf[11],
            )
            if ctf.size > 12:
                row["CCCC"] = ctf[12]
            else:
                row["CCCC"] = 0
            if ctf.size > 13:
                row["Counts"] = ctf[13]
            else:
                row["Counts"] = 0
        else:
            row["X"] = row["Y"] = row["Z"] = 0
            row["DF"] = row["CC"] = row["DF1"] = row["DF2"] = row["Angast"] = row[
                "CCC"
            ] = row["CCCC"] = row["Counts"] = 0
            row["DF1-DF2"] = row["CC_High_Res"] = 0
            row["Pixel size"] = row["Voltage"] = row["Magnification"] = 0
        ctfprofile = "ctf/%s_CTFprof.txt" % i
        if os.path.isfile(ctfprofile):
            profile = np.genfromtxt(
                ctfprofile, dtype="f", delimiter=[6, 14, 14, 14, 14, 14]
            )
            area_in_range = profile[
                np.logical_and(profile[:, 1] >= 0.1, profile[:, 1] <= 0.2)
            ][:, 2]
            if area_in_range.size > 0:
                row["Area from .1 to .2"] = area_in_range.mean()
            else:
                row["Area from .1 to .2"] = 0
            area_in_range = profile[
                np.logical_and(profile[:, 1] >= 0.2, profile[:, 1] <= 0.3)
            ][:, 2]
            if area_in_range.size > 0:
                row["CC_High_Res"] = area_in_range.mean()
            else:
                row["CC_High_Res"] = 0
        else:
            row["Area from .1 to .2"] = row["CC_High_Res"] = 0
        boxfile = "box/" + i + ".box"
        if os.path.isfile(boxfile):
            boxes = np.loadtxt(boxfile, ndmin=2)
            row["Particles"] = boxes.shape[0]
        else:
            row["Particles"] = 0
        xffile = "ali/" + i + ".xf"
        if os.path.isfile(xffile):
            drift = np.loadtxt(xffile)
            if drift.ndim > 1:
                row["Avg_Drift"] = np.average(np.hypot(drift[:, 4], drift[:, 5]))
                row["Max_Drift"] = np.fabs(drift[:, -2:]).max()
            else:
                row["Avg_Drift"] = row["Max_Drift"] = 0
        else:
            row["Avg_Drift"] = row["Max_Drift"] = 0
        csvwriter.writerow(row)
    f.close()


def use_existing_alignments(parameters, new_name):
    new_par_file = "frealign/%s_01.par" % new_name
    shutil.move(new_par_file, new_par_file + "o")

    if not "t" in parameters["ctf_use_lcl"].lower():
        symlink_relative(parameters["class_par"], new_par_file)
    else:
        # assemble new par file with most current set of defocuses
        com = "cat {0} | grep C > {1}".format(parameters["class_par"], new_par_file)
        local_run.run_shell_command(com)
        com = "cat {0} | grep -v C | cut -c1-65 > orientations".format(
            parameters["class_par"]
        )
        local_run.run_shell_command(com)
        com = "cat {0} | grep -v C | cut -c92-136 > prs".format(parameters["class_par"])
        local_run.run_shell_command(com)
        com = "cat {0} | grep -v C | cut -c66-91 > defocuses".format(new_par_file + "o")
        local_run.run_shell_command(com)
        com = 'paste -d "" orientations defocuses prs >> {0}'.format(new_par_file)
        local_run.run_shell_command(com)
        com = "rm orientations defocuses prs"
        local_run.run_shell_command(com)


def get_new_input_list(parameters, inputlist):
    newinput_dict = {}
    if "spr" in parameters["data_mode"]:
        is_spr = True
    else:
        is_spr = False

    spr_pick = (
                is_spr 
                and "detect_rad" in parameters and parameters["detect_rad"] > 0
                and not ("none" in parameters["detect_method"] or "pyp-train" in parameters["detect_method"])
                )
    tomo_vir_pick = (
                    "tomo_vir_rad" in parameters and parameters["tomo_vir_rad"] > 0
                    and not ("none" in parameters["tomo_vir_detect_method"] or "none" in parameters["tomo_vir_method"] or "pyp-train" in parameters["tomo_vir_method"])
                    )
    tomo_spk_pick = (
                    "tomo_spk_rad" in parameters and parameters["tomo_spk_rad"] > 0
                        and not ("none" in parameters["tomo_spk_method"] or "pyp-train" in parameters["tomo_spk_method"])
                    )

    if spr_pick or tomo_vir_pick or tomo_spk_pick:

        for sname in inputlist:

            metadata = pyp_metadata.LocalMetadata(f"./pkl/{sname}.pkl", is_spr=is_spr)
            if "box" in metadata.data.keys():
                boxx = metadata.data["box"].to_numpy()
                boxx_exists = boxx.size > 0
            else:
                boxx_exists = False

            box_size = 0

            if boxx_exists:
                if is_spr:
                    box = boxx[
                        np.logical_and(
                            boxx[:, 4] == 1, boxx[:, 5] >= int(parameters["extract_cls"])
                        )
                    ]
                    box_size = box.shape[0]
                else:
                    box_size = boxx.shape[0]

            if not boxx_exists:
                logger.warning(f"boxx file not found. Box size = {box_size}. Removing {sname} from list")
            else:
                newinput_dict.update({sname:box_size})
        newinputlist = sorted(newinput_dict, key=newinput_dict.get, reverse=True)
        logger.warning("Updating films order to reflect the particles number rank")
    else:
        newinputlist = inputlist

    return newinputlist


def get_non_empty_lines_from_par(par_file):
    lines = [
        _f
        for _f in (line.rstrip() for line in open(par_file) if not line.startswith("C"))
        if _f
    ]
    return lines


def get_particles_from_par(par_file):
    lines = get_non_empty_lines_from_par(par_file)
    return int(lines[-1].split()[0])


def get_particles_from_dataset(dataset):
    par_file = "{}_01.par".format(dataset)
    return get_particles_from_par(par_file)


def create_curr_iter_par(iteration, name, previous, current, is_frealignx):
    with open("scratch/%s_%02d.par" % (name, iteration - 1), "w") as f:
        if is_frealignx:
            last_column = 144
        else:
            last_column = 136
        for i in open(previous + ".par"):
            if i.startswith("C"):
                # pass
                f.write(i)
            else:
                if len(i) < last_column:
                    f.write(i[:last_column])
                else:
                    f.write(i[:last_column] + "\n")
    shutil.copy(
        "scratch/%s_%02d.par" % (name, iteration - 1), "scratch/" + current + ".par"
    )


def csp_spr_swarm(filename, parameters, only_inside=False, csp_swarm=False):
    """Extract per-frame box and parx for SPR.

    Parameters
    ----------
    filename : str, Path
        Movie filename
    parameters : dict
        Main configurations taken from .pyp_config
    only_inside : bool, optional
        Whether to only extract particles inside image boundary, by default False
    csp_swarm : bool, optional
        Whether to use local trajectories saved from in ali/, by default False

    Returns
    ----------
    allboxes
        Box array for all frames in all particles
    allparxs
        Parx strings for all frames in all particles
    """

    # read extended box file
    if os.path.exists("box/{}.boxx".format(filename)):
        boxx = np.loadtxt("box/{}.boxx".format(filename), ndmin=2)
        box = boxx[
            np.logical_and(
                boxx[:, 4] == 1, boxx[:, 5] >= int(parameters["extract_cls"])
            )
        ]
        # box = boxx[ boxx[:,5] >= int(parameters['extract_cls'] ) ]
    else:
        box = np.loadtxt("box/{}.box".format(filename), ndmin=2)

    ctf = np.loadtxt("ctf/{}.ctf".format(filename))
    dims = np.array([ctf[6], ctf[7]])
    xf = np.loadtxt("ali/{}.xf".format(filename), ndmin=2)

    allboxes = []
    allparxs = []

    if not box.size > 0:
        return allboxes, allparxs

    refinement = project_params.resolve_path(parameters["refine_parfile"])

    if Path(refinement).exists() and not "relion_frames" in parameters["extract_fmt"]:
        # find zero-indexed film number for this micrograph
        series = project_params.get_film_order(parameters, filename) - 1

        ref = np.array(
            [
                line.split()
                for line in open(refinement)
                if not line.startswith("C") and line.split()[7] == "{}".format(series)
            ],
            dtype=float,
        )

        if ref.shape[-1] > 13:
            scores = True
        else:
            scores = False

        if box.shape[0] != ref.shape[0]:
            raise Exception(
                "Number of particle images and parameters do not match: {0} != {1}".format(
                    box.shape[0], ref.shape[0]
                )
            )

    boxsize = int(parameters["extract_box"]) * int(parameters["extract_bin"])

    pixel = (
        float(parameters["scope_pixel"])
        * float(parameters["data_bin"])
        * float(parameters["extract_bin"])
    )

    wgh = parameters["scope_wgh"]
    cs = parameters["scope_cs"]
    voltage = parameters["scope_voltage"]
    binning = float(parameters["data_bin"]) * float(parameters["extract_bin"])
    magnification = ctf[11] / binning
    dstep = float(pixel) * magnification / 10000.0
    ccc = ctf[5]

    if "ctf_use_ast" in parameters.keys() and not parameters["ctf_use_ast"]:
        df1 = df2 = ctf[0]  # TOMOCTFFIND
        angast = 45.0
    else:
        df1 = ctf[2]  # CTFFIND3
        df2 = ctf[3]  # CTFFIND3
        angast = ctf[4]  # CTFFIND3

    # global .parx parameters
    dose = tilt = ppsi = ptheta = pphi = 0
    norm0 = norm1 = norm2 = 0
    a03 = a07 = a11 = a12 = a13 = a14 = 0
    a15 = 1

    # for frame in range(xf.shape[0]):
    last = parameters["movie_last"]
    z = xf.shape[0]
    if last >= 0 and last <= z:
        z = last
    local_frame = 0
    global_counter = 0

    for frame in range(parameters["movie_first"], z):
        # atan2( a21 - a12, a22 + a11 )
        axis = math.degrees(
            math.atan2(xf[frame][2] - xf[frame][1], xf[frame][3] + xf[frame][0])
        )

        # current's frame transformation
        transformation = np.linalg.inv(
            np.vstack([xf[frame].take([0, 1, 4]), xf[frame].take([2, 3, 5]), [0, 0, 1]])
        )
        local_particle = 0

        for particle in range(box.shape[0]):
            # particle's coordinates with respect to center of micrograph
            coordinates = np.append(
                box[particle, 0:2] + box[particle, 3] / 2 - dims / 2.0, 1
            )

            # correct for local drift if available
            if csp_swarm:
                # use local drifts saved in ali instead
                local_drifts = (
                    Path("ali/local_drifts") / f"{filename}_P{particle:04d}_frames.xf"
                )
            else:
                local_drifts = (
                    Path(os.environ["PYP_SCRATCH"])
                    / filename
                    / f"{filename}_P{particle:04d}_frames.xf"
                )

            if os.path.exists(local_drifts):

                if particle == 0 and frame == int(parameters["movie_first"]):
                    logger.info("Using local alignments %s", local_drifts)

                xf_local = np.loadtxt(local_drifts, ndmin=2)

                # current's frame transformation
                # XD: if local, rounds the global translations first, check if frame refinement does the same
                transformation = np.linalg.inv(
                    np.vstack(
                        [
                            np.round(xf[frame]).take([0, 1, 4])
                            + [0, 0, xf_local[frame][4]],
                            np.round(xf[frame]).take([2, 3, 5])
                            + [0, 0, xf_local[frame][5]],
                            [0, 0, 1],
                        ]
                    )
                )
            else:
                # current's frame transformation
                transformation = np.linalg.inv(
                    np.vstack(
                        [
                            xf[frame].take([0, 1, 4]),
                            xf[frame].take([2, 3, 5]),
                            [0, 0, 1],
                        ]
                    )
                )

            # transformed coordinates in current frame (boxer format)
            pos = transformation.dot(coordinates)[0:2] + dims / 2.0 - boxsize / 2.0
            # logger.info("transformation", transformation)
            # logger.info("coordinates", coordinates)

            box_pos = pos.round()
            # because can only extract particles in integer coordinates, account for the box error when inputting parx
            box_error = pos - box_pos

            # check if new box is contained in micrograph
            if (
                box_pos[0] < 0
                or box_pos[1] < 0
                or box_pos[0] >= dims[0] - boxsize
                or box_pos[1] >= dims[1] - boxsize
            ):
                # print dims, boxsize
                """
                logger.info(
                    "Particle {3} = [ {0}, {1} ] falls outside frame {2} dimensions".format(
                        box_pos[0], box_pos[1], local_frame, particle
                    )
                )
                """
                if only_inside:
                    logger.info("Skipping particle frame.")
                    continue
            if (
                "relion_frames" in parameters["extract_fmt"].lower()
                or "local" in parameters["extract_fmt"].lower()
            ):
                # relion-1.3
                # f.write("""data_\nloop_\n_rlnMicrographName #1\n_rlnCoordinateX #2\n_rlnCoordinateY #3\n_rlnImageName #4\n_rlnDefocusU #5\n_rlnDefocusV #6\n_rlnDefocusAngle #7\n_rlnVoltage #8\n_rlnSphericalAberration #9\\n_rlnAmplitudeContrast #10\n_rlnMagnification #11\n_rlnDetectorPixelSize #12\n_rlnCtfFigureOfMerit #13\n_rlnParticleName #14\n""")
                frame_name = "%.6i@%s/relion/%s_frames.mrc" % (
                    local_frame + 1,
                    os.getcwd(),
                    filename,
                )
                micrograph_name = (
                    "%.6i@%s/relion/Particles/Micrographs/%s_frames_particles.mrcs"
                    % (global_counter + 1, os.getcwd(), filename)
                )
                micrograph_name = (
                    "%.6i@Particles/Micrographs/%s_frames_particles.mrcs"
                    % (global_counter + 1, filename)
                )
                box_x = box[particle, 0] + int(parameters["extract_box"]) / 2
                box_y = box[particle, 1] + int(parameters["extract_box"]) / 2
                allboxes.append([box_pos[0], box_pos[1], local_frame])

                # retrieve Psi angle from RELION refinement
                angle_psi = 0

                # set dummy value for this field
                autopick_figure_of_merit = 1.0

                # get class number from RELION refinement
                class_number = 0
                allparxs.append(
                    "%d %13.6f %13.6f %s %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %13.6f %d %s\n"
                    % (
                        local_particle + 1,
                        box_x / binning,
                        box_y / binning,
                        micrograph_name,
                        df1,
                        df2,
                        angast,
                        float(voltage),
                        float(cs),
                        float(wgh),
                        magnification,
                        dstep,
                        ccc,
                        angle_psi,
                        autopick_figure_of_merit,
                        class_number,
                        ("%06i@" % (local_frame + 1))
                        + "Micrographs/"
                        + filename
                        + "_frames.mrcs",
                    )
                )
                """
                _rlnMicrographName #1
                _rlnCoordinateX #2
                _rlnCoordinateY #3
                _rlnImageName #4
                _rlnDefocusU #5
                _rlnDefocusV #6
                _rlnDefocusAngle #7
                _rlnVoltage #8
                _rlnSphericalAberration #9
                _rlnAmplitudeContrast #10
                _rlnMagnification #11
                _rlnDetectorPixelSize #12
                _rlnCtfFigureOfMerit #13
                _rlnAnglePsi #14
                _rlnAutopickFigureOfMerit #15
                _rlnClassNumber #16
                _rlnOriginalParticleName #17
                """

            # Write out using FREALIGN .par format
            elif (
                not scores
                and ref[particle][11] < float(parameters["csp_thresh"])
                or scores
                and ref[particle][14] > float(parameters["csp_thresh"])
            ):
                psi, the, phi = np.radians(ref[particle][1:4])

                a00 = cos(phi) * cos(the) * cos(psi) - sin(phi) * sin(psi)
                a01 = -sin(phi) * cos(the) * cos(psi) + cos(phi) * sin(psi)
                a02 = -sin(the) * cos(psi)

                a04 = cos(phi) * cos(the) * sin(psi) + sin(phi) * cos(psi)
                a05 = -sin(phi) * cos(the) * sin(psi) + cos(phi) * cos(psi)
                a06 = -sin(the) * sin(psi)

                a08 = sin(the) * cos(phi)
                a09 = -sin(the) * sin(phi)
                a10 = cos(the)

                if not scores:
                    (
                        count,
                        psi,
                        the,
                        phi,
                        sx,
                        sy,
                        mag,
                        film,
                        df1,
                        df2,
                        angast,
                        presa,
                        dpres,
                    ) = ref[particle][:]
                    occ = 100
                    sigma = 0.5
                    logp = score = change = 0
                else:
                    (
                        count,
                        psi,
                        the,
                        phi,
                        sx,
                        sy,
                        mag,
                        film,
                        df1,
                        df2,
                        angast,
                        occ,
                        logp,
                        sigma,
                        score,
                        change,
                    ) = ref[particle][:16]

                # correct for .box quantization error
                if csp_swarm:
                    xshift, yshift = sx + box_error[0], sy + box_error[1]
                else:
                    xshift, yshift = sx, sy

                scan_order, confidence, ptl_CCX = (
                    local_frame,
                    -local_frame,
                    local_frame + 1,
                )

                # frealign_v8
                #           PSI   THETA     PHI     SHX     SHY     MAG  FILM      DF1      DF2  ANGAST  PRESA   DPRES

                # allparxs.append( '%8.2f%8.2f%8.2f%8.2f%8.2f%8.0f%6d%9.1f%9.1f%8.2f%7.2f%8.2f%9d%9.2f%9.2f%9d%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f%9.2f' % (psi, the, phi, xshift, yshift, mag, film, df1, df2, angast, presa, dpres, local_particle, tilt, dose, scan_order, confidence, ptl_CCX, axis, norm0, norm1, norm2, a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15, ppsi, ptheta, pphi ) )

                if True:
                    allboxes.append([box_pos[0], box_pos[1], local_frame])
                else:
                    # apply shifts before particle extraction
                    newx = np.round(sx / pixel)
                    newy = np.round(sy / pixel)
                    sx = sx / pixel - newx
                    sy = sy / pixel - newy
                    allboxes.append([box_pos[0] + newx, box_pos[1] + newy, local_frame])

                # frealign_v9
                #           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC     -LogP      SIGMA   SCORE  CHANGE
                allparxs.append(
                    frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE_WO_NO
                    % (
                        psi,
                        the,
                        phi,
                        xshift,
                        yshift,
                        mag,
                        film,
                        df1,
                        df2,
                        angast,
                        occ,
                        logp,
                        sigma,
                        score,
                        change,
                        local_particle,
                        tilt,
                        dose,
                        scan_order,
                        confidence,
                        ptl_CCX,
                        axis,
                        norm0,
                        norm1,
                        norm2,
                        a00,
                        a01,
                        a02,
                        a03,
                        a04,
                        a05,
                        a06,
                        a07,
                        a08,
                        a09,
                        a10,
                        a11,
                        a12,
                        a13,
                        a14,
                        a15,
                        ppsi,
                        ptheta,
                        pphi,
                    )
                )
            local_particle += 1
            global_counter += 1
        local_frame += 1

    return allboxes, allparxs


def tomo_load_frame_xf(parameters, name, xf_path="."):
    """ Load all frame transformation files (.xf) for a tilt-series  

    Parameters
    ----------
    parameters : dict
        PYP parameter
    name : str
        Name of the tilt-series

    Returns
    -------
    list[np.array]
        List contains number of 2d array (transformation each tilt)
    list[str]
        List contains filenames of frame transformation files
    """
    root_pattern, file_format = os.path.splitext(str(parameters["movie_pattern"]))

    # get the position of tilt angle in the filenames
    try:
        pos_tiltangle = root_pattern.replace("TILTSERIES", "").split("_").index("ANGLE")
    except:
        pos_tiltangle = -1

    regex = movie2regex(
        str(parameters["movie_pattern"]).replace(file_format, ".xf"), name
    )
    r = re.compile(regex)

    # search all individual movie frames and sort them based on the tilt angle (negative -> positive)
    xf_files = [f for f in list(filter(r.match, os.listdir(xf_path)))]

    if not len(xf_files) > 0:
        raise Exception("Cannot find any tilt movies transformation files (.xf)")

    # sort based on tilt angles
    xf_files = sorted(
        xf_files,
        key=lambda x: float(
            x.replace(name, "").replace(".xf", "").split("_")[pos_tiltangle]
        ),
    )

    xf_frames = [np.loadtxt(os.path.join(xf_path, f), ndmin=2) for f in xf_files]

    return xf_frames, xf_files


def get_max_resolution(
    filename, path="."
):
    """Extract per-frame box and parx for SPR.

    Parameters
    ----------
    filename : str, Path
        Movie filename
    parameters : dict
        Main configurations taken from .pyp_config
    only_inside : bool, optional
        Whether to only extract particles inside image boundary, by default False
    use_frames : bool, optional
        Whether to use frames for coordinate extraction, by default False
    use_existing_frame_alignments : bool, optional
        Whether to use local trajectories saved from in ali/, by default False

    Returns
    ----------
    Estimated resolution
        Resolution in A
    """

    pkl = os.path.join(path, filename + ".pkl")
    assert (os.path.exists(pkl)), f"{pkl} does not exist, please re-run sprswarm"
    metadata_object = pyp_metadata.LocalMetadata(pkl)
    metadata = metadata_object.data

    for key in ["ctf"]:
        assert (key in metadata), f"{key} is not included in {pkl}, please re-run sprswarm"

    return metadata["ctf"].to_numpy()[12]


def spa_extract_coordinates(
    filename, parameters, only_inside, use_frames, use_existing_frame_alignments, path="."
):
    """Extract per-frame box and parx for SPR.

    Parameters
    ----------
    filename : str, Path
        Movie filename
    parameters : dict
        Main configurations taken from .pyp_config
    only_inside : bool, optional
        Whether to only extract particles inside image boundary, by default False
    use_frames : bool, optional
        Whether to use frames for coordinate extraction, by default False
    use_existing_frame_alignments : bool, optional
        Whether to use local trajectories saved from in ali/, by default False

    Returns
    ----------
    allboxes
        Box array for all frames in all particles
    allparxs
        Parx strings for all frames in all particles
    """

    pkl = os.path.join(path, filename + ".pkl")
    assert (os.path.exists(pkl)), f"{pkl} does not exist, please re-run sprswarm"
    metadata_object = pyp_metadata.LocalMetadata(pkl)
    metadata = metadata_object.data

    for key in ["image", "box", "ctf", "drift"]:
        assert (key in metadata), f"{key} is not included in {pkl}, please re-run sprswarm"

    # read extended box file
    # if os.path.exists( os.path.join(path, "{}.boxx".format(filename))):
    #     boxx = np.loadtxt( os.path.join(path,"{}.boxx".format(filename)), ndmin=2)
    #     box = boxx[
    #         np.logical_and(boxx[:, 4] == 1, boxx[:, 5] >= parameters["extract_cls"])
    #     ]
    #     # box = boxx[ boxx[:,5] >= int(parameters['extract_cls'] ) ]
    # else:
    #     box = np.loadtxt(os.path.join(path,"{}.box".format(filename)), ndmin=2)

    boxx = metadata["box"].to_numpy()
    box = boxx[
        np.logical_and(boxx[:, 4] == 1, boxx[:, 5] >= parameters["extract_cls"])
    ]

    ctf = metadata["ctf"].to_numpy() # np.loadtxt(os.path.join(path,"{}.ctf".format(filename)))
    dims = np.array([ctf[6][0], ctf[7][0]])
    xf = metadata["drift"].to_numpy() # np.loadtxt(os.path.join(path,"{}.xf".format(filename)), ndmin=2)

    allboxes = []
    allparxs = []
    allparxs.append([])

    if not box.size > 0:
        logger.warning("You have empty partilce coordinates, return empty parfile")
        return allboxes, allparxs

    scores = False

    if "refine_parfile" in parameters.keys():
        refinement = project_params.resolve_path(parameters["refine_parfile"])
    else:
        refinement = "none"

    if (
        "refine_parfile" in parameters.keys()
        and Path(refinement).exists()
        and not "relion_frames" in parameters["extract_fmt"]
    ):
        # find zero-indexed film number for this micrograph
        series = project_params.get_film_order(parameters, filename) - 1

        ref = np.array(
            [
                line.split()
                for line in open(refinement)
                if not line.startswith("C") and line.split()[7] == "{}".format(series)
            ],
            dtype=float,
        )

        if ref.shape[-1] > 13:
            scores = True

        if box.shape[0] != ref.shape[0]:
            raise Exception(
                "Number of particles and parameters do not match: {0} != {1}".format(
                    box.shape[0], ref.shape[0]
                )
            )
    else:
        ref = None
   
    boxsize = int(parameters["extract_box"]) * int(parameters["extract_bin"])

    pixel = (
        float(parameters["scope_pixel"])
        * float(parameters["data_bin"])
        * float(parameters["extract_bin"])
    )

    wgh = parameters["scope_wgh"]
    cs = parameters["scope_cs"]
    voltage = parameters["scope_voltage"]
    binning = float(parameters["data_bin"]) * float(parameters["extract_bin"])
    magnification = ctf[11]
    dstep = float(pixel) * magnification / 10000.0
    ccc = ctf[5]

    if "ctf_use_ast" in parameters.keys() and not parameters["ctf_use_ast"]:
        df1 = df2 = ctf[0]  # TOMOCTFFIND
        angast = 45.0
    else:
        df1 = ctf[2]  # CTFFIND3
        df2 = ctf[3]  # CTFFIND3
        angast = ctf[4]  # CTFFIND3

    # global .parx parameters
    dose = tilt = ppsi = ptheta = pphi = 0
    norm0 = norm1 = norm2 = 0
    a00 = a05 = a10 = a15 = 1
    a01 = a02 = a04 = a06 = a08 = a09 = 0
    a03 = a07 = a11 = a12 = a13 = a14 = 0

    sx = sy = 0
    mag = magnification
    film = 0
    presa = dpres = 0
    occ = 100
    sigma = 0.5
    logp = change = 0
    score = 0.5
    # for frame in range(xf.shape[0]):
    last = parameters["movie_last"]
    z = xf.shape[0]
    if last >= 0 and last <= z:
        z = last
    local_frame = 0
    global_counter = 0
    # for micrograph input
    if parameters["movie_first"] == z:
        z += 1
    for frame in range(parameters["movie_first"], z):

        # atan2( a21 - a12, a22 + a11 )
        axis = math.degrees(
            math.atan2(xf[frame][2] - xf[frame][1], xf[frame][3] + xf[frame][0])
        )

        # current's frame transformation
        transformation = np.linalg.inv(
            np.vstack([xf[frame].take([0, 1, 4]), xf[frame].take([2, 3, 5]), [0, 0, 1]])
        )
        local_particle = 0

        for particle in range(box.shape[0]):

            # particle's coordinates with respect to center of micrograph
            coordinates = np.append(
                box[particle, 0:2] + box[particle, 3] / 2 - dims / 2.0, 1
            )

            # correct for local drift if available
            if use_existing_frame_alignments:
                # use local drifts saved in ali instead
                local_drifts = "ali/local_drifts/{0}_P{1}_frames.xf".format(
                    filename, "%04d" % particle
                )
            else:
                local_drifts = "{0}/{1}/{1}_P{2}_frames.xf".format(
                    os.environ["PYP_SCRATCH"], filename, "%04d" % particle
                )

            if os.path.exists(local_drifts):

                if particle == 0 and frame == int(parameters["movie_first"]):
                    logger.info("Using local alignments %s", local_drifts)

                xf_local = np.loadtxt(local_drifts, ndmin=2)

                # current's frame transformation
                # XD: if local, rounds the global translations first, check if frame refinement does the same
                transformation = np.linalg.inv(
                    np.vstack(
                        [
                            np.round(xf[frame]).take([0, 1, 4])
                            + [0, 0, xf_local[frame][4]],
                            np.round(xf[frame]).take([2, 3, 5])
                            + [0, 0, xf_local[frame][5]],
                            [0, 0, 1],
                        ]
                    )
                )
            else:
                # current's frame transformation
                transformation = np.linalg.inv(
                    np.vstack(
                        [
                            xf[frame].take([0, 1, 4]),
                            xf[frame].take([2, 3, 5]),
                            [0, 0, 1],
                        ]
                    )
                )

            if use_frames:
                # transformed coordinates in current frame (boxer format)
                pos = transformation.dot(coordinates)[0:2]
            else:
                pos = coordinates[0:2]

            pos = pos + dims / 2.0  # - boxsize / 2.0
            box_pos = pos.round()

            # because can only extract particles in integer coordinates, account for the box error when inputting parx
            box_error = pos - box_pos

            # check if new box is contained in micrograph
            if (
                box_pos[0] - (boxsize / 2.0) < 0
                or box_pos[1] - (boxsize / 2.0) < 0
                or box_pos[0] >= dims[0] - (boxsize / 2.0)
                or box_pos[1] >= dims[1] - (boxsize / 2.0)
            ):
                # print dims, boxsize
                """
                logger.info(
                    "Particle {3} = [ {0}, {1} ] falls outside frame {2} dimensions".format(
                        box_pos[0], box_pos[1], local_frame, particle
                    )
                )
                """
                if only_inside:
                    logger.info("Skipping particle frame.")
                    continue
            if (
                ref is None
                or not scores
                and ref[particle][11] < parameters["csp_thresh"]
                or scores
                and ref[particle][14] >= parameters["csp_thresh"]
            ):

                if ref is None:
                    psi, the, phi = 0, 0, 0
                else:
                    # HF - preserve identity matrix for later frame alignment 
                    pass
                    """
                    psi, the, phi = np.radians(ref[particle][1:4])

                    a00 = cos(phi) * cos(the) * cos(psi) - sin(phi) * sin(psi)
                    a01 = -sin(phi) * cos(the) * cos(psi) + cos(phi) * sin(psi)
                    a02 = -sin(the) * cos(psi)

                    a04 = cos(phi) * cos(the) * sin(psi) + sin(phi) * cos(psi)
                    a05 = -sin(phi) * cos(the) * sin(psi) + cos(phi) * cos(psi)
                    a06 = -sin(the) * sin(psi)

                    a08 = sin(the) * cos(phi)
                    a09 = -sin(the) * sin(phi)
                    a10 = cos(the)
                    """

                if ref is not None:
                    if not scores:
                        (
                            count,
                            psi,
                            the,
                            phi,
                            sx,
                            sy,
                            mag,
                            film,
                            df1,
                            df2,
                            angast,
                            presa,
                            dpres,
                        ) = ref[particle][:13]
                    else:
                        (
                            count,
                            psi,
                            the,
                            phi,
                            sx,
                            sy,
                            mag,
                            film,
                            df1,
                            df2,
                            angast,
                            occ,
                            logp,
                            sigma,
                            score,
                            change,
                        ) = ref[particle][:16]

                # correct for .box quantization error
                if use_existing_frame_alignments:
                    xshift, yshift = sx + box_error[0], sy + box_error[1]
                else:
                    xshift, yshift = sx, sy

                scan_order, confidence, ptl_CCX = (
                    local_frame,
                    -local_frame,
                    local_frame + 1,
                )

                allboxes.append([box_pos[0], box_pos[1], local_frame])

                if False and "cc" in project_params.param(
                    parameters["refine_metric"], iteration=2
                ):

                    """
                    C FREALIGN NEW parameter file
                    C     1       2       3       4         5         6       7     8        9       10      11      12        13         14      15      16
                    C    NO     PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LOGP      SIGMA   SCORE  CHANGE
                    """
                    allparxs[0].append(
                        frealign_parfile.EXTENDED_CCLIN_PAR_STRING_TEMPLATE_WO_NO
                        % (
                            psi,
                            the,
                            phi,
                            xshift,
                            yshift,
                            mag,
                            film,
                            df1,
                            df2,
                            angast,
                            occ,
                            logp,
                            sigma,
                            score,
                            change,
                            local_particle,
                            tilt,
                            dose,
                            scan_order,
                            confidence,
                            ptl_CCX,
                            axis,
                            norm0,
                            norm1,
                            norm2,
                            a00,
                            a01,
                            a02,
                            a03,
                            a04,
                            a05,
                            a06,
                            a07,
                            a08,
                            a09,
                            a10,
                            a11,
                            a12,
                            a13,
                            a14,
                            a15,
                            ppsi,
                            ptheta,
                            pphi,
                        )
                    )

                else:
                    """C     1       2       3       4         5         6       7     8        9       10      11      12        13         14      15      16       17        18        19        20        21        22        23        24        25        26        27        28        29        30        31        32        33        34        35        36        37        38        39        40        41        42        43        44        45"""
                    """C    NO     PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LOGP      SIGMA   SCORE  CHANGE   PTLIND    TILTAN    DOSEXX    SCANOR    CNFDNC    PTLCCX      AXIS     NORM0     NORM1     NORM2  MATRIX00  MATRIX01  MATRIX02  MATRIX03  MATRIX04  MATRIX05  MATRIX06  MATRIX07  MATRIX08  MATRIX09  MATRIX10  MATRIX11  MATRIX12  MATRIX13  MATRIX14  MATRIX15      PPSI    PTHETA      PPHI"""

                    # frealign_v9
                    #           PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC     -LogP      SIGMA   SCORE  CHANGE
                    allparxs[0].append(
                        frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE_WO_NO
                        % (
                            psi,
                            the,
                            phi,
                            xshift,
                            yshift,
                            mag,
                            film,
                            df1,
                            df2,
                            angast,
                            occ,
                            logp,
                            sigma,
                            score,
                            change,
                            local_particle,
                            tilt,
                            dose,
                            scan_order,
                            confidence,
                            ptl_CCX,
                            axis,
                            norm0,
                            norm1,
                            norm2,
                            a00,
                            a01,
                            a02,
                            a03,
                            a04,
                            a05,
                            a06,
                            a07,
                            a08,
                            a09,
                            a10,
                            a11,
                            a12,
                            a13,
                            a14,
                            a15,
                            ppsi,
                            ptheta,
                            pphi,
                        )
                    )
            local_particle += 1
            global_counter += 1

        # if not using frames, we are done
        if not use_frames:
            break

        local_frame += 1

    return allboxes, allparxs


@timer.Timer(
    "csp_extract_coordinates", text="Reading and converting coordinates took: {}", logger=logger.info
)
def csp_extract_coordinates(
    filename,
    parameters,
    working_path,
    current_path,
    skip,
    only_inside=False,
    use_frames=False,
    use_existing_frame_alignments=False,
):
    frame_tag = ""
    if use_frames:
        frame_tag = "_local"

    micrographs = f"{parameters['data_set']}.films"
    with open(micrographs) as f:
        micrograph_list = [line.strip() for line in f]
    if filename in micrograph_list:
        micrograph_index = micrograph_list.index(filename)
        film_col = 7
    else:
        raise Exception(
            "{} is not in {}".format(filename,micrographs)
        )

    if not skip and project_params.csp_is_done(
        os.path.join(working_path, filename), use_frames
    ):
        logger.info(
            "Skipping coordinate extraction (using existing frame coordinates)"
        )

        with timer.Timer(
            "read_boxfiles", text = "Reading allboxes, boxxs took: {}", logger=logger.info
        ):
            # retrieve from files
            allboxes = (
                np.loadtxt(
                    os.path.join(working_path, filename + frame_tag + ".allboxes"), ndmin=2
                )
                .astype(int)
                .tolist()
            )

        allparxs = []
        iteration = parameters["refine_iter"]
        if iteration == 2:
            classes = 1
        else:
            classes = int(project_params.param(parameters["class_num"], iteration))
        for current_class in range(classes):

            # now get latest metadata from merged par file in frealign/maps folder
            if iteration == 2:
                merged_par_file = project_params.resolve_path(
                    parameters["refine_parfile"]
                ) if "refine_parfile" in parameters else "none"
            else:
                if classes > 1:
                    # we want to read the occ updated .par instead of the original compressed par
                    merged_par_file = os.path.join(
                        os.getcwd(),
                        "frealign",
                        "maps",
                        "%s_r%02d_%02d.par"
                        % (parameters["data_set"], current_class + 1, iteration - 1),
                    ) 
                else:
                    merged_par_file = os.path.join(
                        os.getcwd(),
                        "frealign",
                        "maps",
                        "%s_r%02d_%02d.par.bz2"
                        % (parameters["data_set"], current_class + 1, iteration - 1),
                    )      
            
            if os.path.exists(merged_par_file) and ".par" in merged_par_file:
                par_alignment = True
            else:
                par_alignment = False
   
            if par_alignment:
                logger.info("Retrieving alignments from " + Path(merged_par_file).name)
                # decompress file if needed
                if ".bz2" in merged_par_file:
                    with timer.Timer(
                        "Decompressing_parfile", text = "Decompressing " + Path(merged_par_file).name + " took: {}", logger=logger.info
                    ):
                        merged_par_file = frealign_parfile.Parameters.decompress_parameter_file(
                            merged_par_file, parameters["slurm_tasks"]
                        )

                with timer.Timer(
                    "read_alignment", text = "Reading alignments from parfile took: {}", logger=logger.info
                ):
                    index_file = os.path.join( current_path, "csp" , "micrograph_particle.index" )
                    assert os.path.exists(index_file), f"Index file is missing: {index_file}"
                    with open(index_file) as f:
                        index_dict = json.load(f)
                    start, end = index_dict[str(micrograph_index)]
                    step = end - start
                    start += 3
                    extracted_rows = np.loadtxt(merged_par_file, dtype=float, comments="C", skiprows=start, max_rows=step, ndmin=2)
                    allparxs.append(extracted_rows[:, 1:])
                    
            # else, fall back to canonical extended parameters
            else:
                allparxs.append([])
                allparxs_file = os.path.join(working_path, filename + ".allparxs")

                if len(allparxs[current_class]) != len(allboxes) and os.path.exists(
                    allparxs_file
                ):
                    logger.info("Loading refined parameters from " + allparxs_file)
                    with open(allparxs_file) as f:
                        for line in f:
                            allparxs[current_class].append(line[:-1])

    else:

        logger.info("Executing coordinate extraction")

        if "refine_parfile" in parameters.keys():
            refinement = project_params.resolve_path(parameters["refine_parfile"])
        else:
            refinement = 'none'

        # decompress file if needed
        refinement = frealign_parfile.Parameters.decompress_parameter_file(
            refinement, parameters["slurm_tasks"]
        )
        parameters["refine_parfile"] = refinement

        if "tomo" in parameters["data_mode"].lower():
            
            # generate new parx file from previous parx (not containing frame)
            if use_frames and (
                refinement.endswith(".par")
                or refinement.endswith(".parx")
                or refinement.endswith(".bz2")
            ):
                # decompress file if needed
                refinement = frealign_parfile.Parameters.decompress_parameter_file(
                    refinement, parameters["slurm_tasks"]
                )
                try:
                    parx_object_no_frames = frealign_parfile.Parameters.from_file(
                        refinement
                    )
                except:
                    raise Exception("Parfile cannot be read.")

                parx_object_no_frames.data = parx_object_no_frames.data[
                    parx_object_no_frames.data[:, film_col] == micrograph_index
                ]

                allboxes = (
                    np.loadtxt(
                        os.path.join(working_path, filename + ".allboxes"), ndmin=2
                    )
                    .astype(int)
                    .tolist()
                )

                metadata = None 
                if os.path.exists(f"pkl/{filename}.pkl"):
                    metadata = pyp_metadata.LocalMetadata(f"pkl/{filename}.pkl").data

                try:
                    scanords = [
                        order for order in np.loadtxt("raw/%s.order" % filename)
                    ]
                except:
                    scanords = [int(_[0]) for _ in metadata["order"].to_numpy()]

                if len(allboxes) != parx_object_no_frames.data.shape[0]:
                    raise Exception(
                        "The allboxes and parxfile DO NOT have the same length. (%d v.s. %d)"
                        % (len(allboxes), parx_object_no_frames.data.shape[0])
                    )
                try:
                    xf_frames, xf_files = tomo_load_frame_xf(
                        parameters, filename, xf_path="mrc/"
                    )
                except:
                    xf_frames = [metadata["drift"][tilt_idx].to_numpy() for tilt_idx in sorted(metadata["drift"].keys())]
                
                # convert short parxfile to long parxfile/allboxes that contains frames
                [
                    allboxes,
                    allparxs,
                ] = frealign_parfile.Parameters.extendParFileWithFrames(
                    parx_object_no_frames, allboxes, xf_frames, parameters, scanords
                )
            else:
                [allboxes, allparxs] = tomo_extract_coordinates(
                    filename, parameters, use_frames, extract_projections=False
                )
            # copy boxes3d and ctf files to local scratch
            shutil.copy2("csp/{}_boxes3d.txt".format(filename), working_path)

        else:

            if use_frames and (
                refinement.endswith(".par") or refinement.endswith(".parx")
            ):
                try:
                    parx_object_no_frames = frealign_parfile.Parameters.from_file(
                        refinement
                    )
                except:
                    raise Exception("Parfile cannot be read.")

                parx_object_no_frames.data = parx_object_no_frames.data[
                    parx_object_no_frames.data[:, film_col] == micrograph_index
                ]

                allboxes = (
                    np.loadtxt(
                        os.path.join(working_path, filename + ".allboxes"), ndmin=2
                    )
                    .astype(int)
                    .tolist()
                )

                metadata = None 
                # same as single particle to check boxx selection
                if os.path.exists(os.path.join(working_path, filename + ".pkl")):
                    metadata = pyp_metadata.LocalMetadata(os.path.join(working_path, filename + ".pkl")).data
                """    
                    boxx = metadata["box"].to_numpy()
                else:
                    boxx_file = os.path.join(working_path, filename + ".boxx")
                    if os.path.exists(boxx_file):
                        boxx = np.loadtxt(boxx_file, ndmin=2)
                
                indexes = np.argwhere(
                    np.logical_and(
                        boxx[:, 4] == 1, boxx[:, 5] >= int(parameters["extract_cls"])
                    )
                )
                if len(allboxes) > len(indexes):
                    allboxes = [allboxes[i[0]] for i in indexes]
                """
                if len(allboxes) != parx_object_no_frames.data.shape[0]:
                    raise Exception(
                        "The allboxes and parxfile DO NOT have the same length. (%d v.s. %d)"
                        % (len(allboxes), parx_object_no_frames.data.shape[0])
                    )

                if os.path.exists(os.path.join(working_path, filename + ".xf")):
                    xf_frames = [np.loadtxt(os.path.join(working_path, filename + ".xf"), ndmin=2)]
                else:
                    xf_frames = [metadata["drift"].to_numpy()]

                # convert short parxfile to long parxfile/allboxes that contains frames
                [
                    allboxes,
                    allparxs,
                ] = frealign_parfile.Parameters.extendParFileWithFrames(
                    parx_object_no_frames, allboxes, xf_frames, parameters
                )

            else:
                logger.info("Creating parfile from scratch")
                [allboxes, allparxs] = spa_extract_coordinates(
                    filename,
                    parameters,
                    only_inside,
                    use_frames,
                    use_existing_frame_alignments,
                    working_path
                )

        # save frame coordinates to file
        np.savetxt(
            os.path.join(working_path, filename + frame_tag + ".allboxes"),
            np.array(allboxes).astype(int),
            fmt="%i",
        )

        # save metadata to file and only initialize for 1 class
        with open(
            os.path.join(working_path, filename + frame_tag + ".allparxs"), "w"
        ) as f:
            f.writelines("%s\n" % item for item in allparxs[0])
        

    return allboxes, allparxs


def tomo_extract_coordinates(
    filename, parameters, use_frames=True, extract_projections=False
):
    """Extract per-frame box and parx for TOMO.

    Parameters
    ----------
    filename : str, Path
        Movie filename
    parameters : dict
        Main configurations taken from .pyp_config
    extract_projections : bool, optional
        TODO, by default False

    Returns
    ----------
    allboxes
        Box array for all frames in all particles
    allparxs
        Parx strings for all frames in all particles
    """
    name = os.path.basename(filename)
    # distance_from_equator = float(parameters["tomo_vir_detect_band"])
    min_tilt = float(parameters["reconstruct_mintilt"])
    max_tilt = float(parameters["reconstruct_maxtilt"])
    refinement = project_params.resolve_path(parameters["refine_parfile"])

    # get metadata from pickle
    pkl = Path("pkl") / f"{name}.pkl"
    assert (os.path.exists(pkl)), f"{pkl} does not exist, please re-run tomoswarm"
    metadata_object = pyp_metadata.LocalMetadata(pkl)
    metadata = metadata_object.data

    # check if we have all required data
    for key in ["image", "tomo", "order", "ali", "tlt", "ctf"]:
        assert (key in metadata), f"{key} is not included in {pkl}, please re-run tomoswarm"

    # Decompress
    refinement = frealign_parfile.Parameters.decompress_parameter_file(
    refinement, parameters["slurm_tasks"]
    )

    cutboxsize = int(parameters["extract_box"]) * int(parameters["extract_bin"])

    # virions are binned by 2 by default, if spk file is provided this will be reset to 1
    subvol_bin = float(parameters["extract_bin"]) if "extract_bin" in parameters else 1
    final_bin = 1.0

    if "tomo_vir_rad" in parameters:
        virion_bin, virion_boxsize = get_vir_binning_boxsize(parameters["tomo_vir_rad"], parameters["scope_pixel"])
    else:
        virion_bin = 1
        virion_boxsize = 0
        
    # set virion box size
    if virion_boxsize > 0:
        virion_boxsize /= virion_bin
    else:
        virion_bin = 1
        virion_boxsize = 0

    # tomogram binning factor with respect to raw micrographs
    micrographsize_x, micrographsize_y = metadata["image"].at[0, "x"], metadata["image"].at[0, "y"] 
    # binning = get_tomo_binning(micrographsize_x, micrographsize_y, int(parameters["tomo_rec_size"]), squared_image=parameters["tomo_rec_square"])
    binning = parameters["tomo_rec_binning"]
    # logger.info(f"Detected binning factor is {binning}")

    pixel = (
        float(parameters["scope_pixel"])
        * float(parameters["data_bin"])
        * float(parameters["extract_bin"])
    )

    actual_pixel = float(parameters["scope_pixel"]) * float(parameters["data_bin"])

    # find zero-indexed film number for this micrograph
    film = project_params.get_film_order(parameters, filename) - 1

    allboxes_3d = []
    allboxes = []
    allparxs = []
    allparxs.append([])
    allimodboxes = []
    
    # get size of full unbinned reconstruction from .rec file
    rec_X, rec_Y, rec_Z = metadata["tomo"].at[0, "x"], metadata["tomo"].at[0, "y"], metadata["tomo"].at[0, "z"]
    recX, recY, recZ = [binning * _ for _ in [rec_X, rec_Y, rec_Z]]

    # Image center per IMOD's convention
    center_X, center_Y, center_Z = [x / 2 for x in [recX, recY, recZ]]
    
    # get some geometry values
    min_micrograph_x = (recX - micrographsize_x) / 2.0
    min_micrograph_y = (recY - micrographsize_y) / 2.0
    max_micrograph_x = recX - min_micrograph_x - cutboxsize
    max_micrograph_y = recY - min_micrograph_y - cutboxsize

    # find zero-tilt image
    tilts = metadata["tlt"].to_numpy()
    zero_tilt_line = np.argmin(abs(tilts))
    logger.info(f"Using image {zero_tilt_line} {tilts[zero_tilt_line]} as zero tilt image")

    scan_order_list = [_[0] for _ in metadata["order"].to_numpy()]
    assert (len(scan_order_list) == len(tilts)), f"Scanning order does not match to tilt angle. {len(scan_order_list)} != {len(tilts)}"

    # Identify manually ignored views
    excluded_views = []
    if "exclude" in metadata:
        [ excluded_views.append(f + 1) for f in metadata["exclude"].to_numpy()[:,-1] ]

    if len(excluded_views) > 0:
        logger.info(f"Views to exclude {excluded_views}")


    image_counter = 1
    global_spike_counter = 0
    
    if "vir" in metadata:
        virion_coordinates = metadata["vir"].to_numpy()
    elif os.path.exists("mod/%s.txt" % name):
        virion_coordinates = np.loadtxt(
            "mod/%s.txt" % name,
            comments="particle_x",
            usecols=(list(range(10))),
            ndmin=2,
        )
    elif "box" in metadata:
        virion_coordinates = metadata["box"].to_numpy()
    else:
        virion_coordinates = np.empty( shape=(0, 0) )
    
    # create xf from metadata
    metadata_object.writeTextFile(metadata["ali"].to_numpy(), "%s.xf")
    inversexf = Path(os.environ["PYP_SCRATCH"]) / f"{name}_inverse.xf"

    command = """
%s/bin/xfinverse << EOF
%s.xf
%s
1
EOF
""" % (
        get_imod_path(),
        name,
        inversexf,
    )
    [output, error] = local_run.run_shell_command(command, verbose=False)

    os.remove(f"{name}.xf")
    inverse_xf_file = np.loadtxt(inversexf, ndmin=2)

    # preload frame motion correction files
    if use_frames:
        xf_frames, xf_files = tomo_load_frame_xf(parameters, name, xf_path=os.path.join(os.environ["PYP_SCRATCH"],name))

    # pre-load magnification matrix
    [output, error] = local_run.run_shell_command(
        "%s/bin/xf2rotmagstr %s" % (get_imod_path(), inversexf),
    )
    xf_rot_mag = output.split("\n")

    # preload 3D refinement file
    alignmentSVA = {}
    with open(refinement) as myfile:
        for line in myfile.readlines():
            particle_name = os.path.basename(line.split()[-1])
            alignmentSVA[particle_name] = line.split()
    
    if os.path.exists(f"mod/{name}_gold.mod"):
        command = "{0}/bin/model2point mod/{1}_gold.mod {2}_gold.txt".format(
            get_imod_path(), name, Path(os.environ["PYP_SCRATCH"]) / name
        )
        local_run.run_shell_command(command)
        fiducials = np.loadtxt(
            Path(os.environ["PYP_SCRATCH"]) / f"{name}_gold.txt", ndmin=2
        )
    else:
        fiducials = np.empty([0])

    # pre-load tilt-series alignment
    tilt_series_alignment = metadata["ali"].to_numpy()

    # read defocus estimation per individual tilt
    defocus_per_tilt = metadata["ctf"].to_numpy()

    # HF Liu: calculate the height of specimen in tomogram using mean z height of all particles
    if virion_coordinates.size > 0:
        z_specimen = (
            recZ
            - binning
            * (max(virion_coordinates[:, 2]) + min(virion_coordinates[:, 2]))
            / 2.0
        )
    else:
        z_specimen = 0
    height_specimen = (z_specimen - (recZ / 2)) * actual_pixel

    logger.info(f"The height of specimen in the tomogram is {height_specimen:.2f} A")

    if Path(refinement).exists() and ".par" in parameters["refine_parfile"]:
        input = frealign_parfile.Parameters.from_file(refinement).data

    # traverse all virions in tilt series
    for vir in range(virion_coordinates.shape[0]):

        if os.path.exists("mod/%s.txt" % name):
            BINNING_FOR_PICKING = 2
            Z_FOR_PICKING = rec_Z
            channel_x, channel_y, channel_z, liposome_x, liposome_y, liposome_z = list(
                [x / BINNING_FOR_PICKING for x in virion_coordinates[vir][0:6]]
            )

            channel_y = (
                channel_y - (Z_FOR_PICKING // (2 * (BINNING_FOR_PICKING))) + (rec_Z // 2)
            )
            virion_coordinates[vir, :3] = [channel_x, channel_z, channel_y]

        vir_x, vir_y, vir_z = [
            binning * virion_coordinates[vir, 0],
            binning * virion_coordinates[vir, 1],
            recZ - binning * virion_coordinates[vir, 2],
        ]

        # check if we have picked spikes for this virion
        virion_file = "sva/%s_vir%04d_cut.txt" % (name, vir)
        if os.path.isfile(virion_file):
            spikes_in_virion = np.loadtxt(
                virion_file, comments="number", usecols=(list(range(32))), ndmin=2
            )
            if spikes_in_virion.shape[0] == 0:
                logger.warning(
                    "File {0} contains no spikes. Skipping".format(virion_file)
                )
                continue
        elif "box" in metadata or os.path.exists("mod/%s.spk" % name) or os.path.exists("mod/%s.txt" % name):
            # use origin if we are using isolated particles
            spikes_in_virion = np.zeros([1, 7])
            virion_bin = 1
        else:
            logger.warning(f"File {virion_file} not found. Skipping.")
            continue


        # for all spikes in current virion
        for spike in range(spikes_in_virion.shape[0]):
            # for spike in range(1):

            # extract local spike coordinates [0-479]
            spike_x, spike_y, spike_z = spikes_in_virion[spike, 3:6]

            # virion boxsize is supposed to be included in coordinates.txt
            virion_boxsize = spikes_in_virion[spike, 6]

            spike_x, spike_y, spike_z = spike_x, (virion_boxsize - spike_y), spike_z

            # compute distance of current spike to virion equator
            dist = abs(spike_z - virion_boxsize / 2)

            # compute global spike coordinates from virus box size
            spike_X = vir_x + (spike_x - virion_boxsize // 2) * virion_bin
            spike_Y = vir_y + (spike_y - virion_boxsize // 2) * virion_bin
            spike_Z = vir_z + (spike_z - virion_boxsize // 2) * virion_bin

            # check if this spike was used in refinement
            spike_string = ""

            # two possible names of extracted subvolumes
            spike_vol = "%s_vir%04d_spk%04d.mrc" % (name, vir, spike)
            particle_vol = "%s_spk%04d.rec" % (name, vir)
            
            if spike_vol in alignmentSVA:
                spike_string = alignmentSVA[spike_vol]
            elif particle_vol in alignmentSVA:
                spike_string = alignmentSVA[particle_vol]
            else:
                logger.debug(
                    "Skipping spike %s_vir%04d_spk%04d without alignments" % (name, spike, vir) 
                )
                allboxes_3d.append([spike_X, spike_Y, spike_Z])
                global_spike_counter += 1
                continue

            # extract euler angles from 3DAVG refinment
            norm0, norm1, norm2 = list(map(float, spike_string[9:12]))
            (
                m00,
                m01,
                m02,
                m03,
                m04,
                m05,
                m06,
                m07,
                m08,
                m09,
                m10,
                m11,
                m12,
                m13,
                m14,
                m15,
            ) = list(map(float, spike_string[12:28]))

            cutOffset = float(spike_string[31])

            # transform to unbinned shifts
            cutOffset *= virion_bin * subvol_bin / float(parameters["extract_bin"])
            m03 *= virion_bin * subvol_bin / float(parameters["extract_bin"])
            m07 *= virion_bin * subvol_bin / float(parameters["extract_bin"])
            m11 *= virion_bin * subvol_bin / float(parameters["extract_bin"])

            # get the transformed 3D location after SVA
            [dx, dy, dz] = getShiftsForRecenter(
                [norm0, norm1, norm2],
                [
                    m00,
                    m01,
                    m02,
                    m03,
                    m04,
                    m05,
                    m06,
                    m07,
                    m08,
                    m09,
                    m10,
                    m11,
                    m12,
                    m13,
                    m14,
                    m15,
                ],
                cutOffset,
            )
            transformed_3d_loc = [
                spike_X + (dx * float(parameters["extract_bin"])),
                spike_Y + (dy * float(parameters["extract_bin"])),
                spike_Z + (dz * float(parameters["extract_bin"])),
            ]

            allboxes_3d.append(transformed_3d_loc)
            
            tilt_image_counter = 1

            # traverse all images in tilt series
            for tilt in tilts:

                # check if tilt angle is valid and within acceptable range
                if tilt < min_tilt or tilt > max_tilt:
                    logger.debug(
                        "Ignoring image at tilt angle {0} outside range [ {1}, {2} ].".format(
                            tilt, min_tilt, max_tilt
                        )
                    )
                    tilt_image_counter += 1
                    continue

                if tilt_image_counter in excluded_views:
                    logger.debug(
                        "Ignoring image at tilt angle {0} excluded during alignment.".format(
                            tilt
                        )
                    )
                    tilt_image_counter += 1
                    continue

                # print '      Processing image %i at %f degrees tilt' % ( tilt_image_counter, tilt )

                # convert to radians
                angle = math.radians(tilt)

                # 2D spike coordinates in current aligned projection image (with respect to image center)
                tilt_x = (spike_X - center_X) * math.cos(angle) + (
                    spike_Z - center_Z
                ) * math.sin(angle)
                tilt_y = spike_Y - center_Y

                # check if too close to gold fiducials
                if fiducials.size > 0:
                    fiducials_in_tilt = fiducials[
                        fiducials[:, 2] == tilt_image_counter - 1
                    ]
                    near_gold = False
                    for i in range(fiducials_in_tilt.shape[0]):
                        # skip if gold fiducial falls withing box used for particle picking
                        if (
                            math.hypot(
                                fiducials_in_tilt[i, 0] * binning - (tilt_x + center_X),
                                fiducials_in_tilt[i, 1] * binning - spike_Y,
                            )
                            < cutboxsize / 2.0
                        ):
                            near_gold = True
                            break

                    if near_gold:
                        logger.debug(
                            "Skipping projection %s too close to gold fiducial" % tilt
                        )
                        tilt_image_counter += 1
                        continue
                
                distance_to_tilt_axis = tilt_x

                T = inverse_xf_file[tilt_image_counter - 1, :6]

                # get parameters from IMOD's affine transformation
                for line in xf_rot_mag:
                    if (
                        "rot=" in line
                        and line.split()[0] == str(tilt_image_counter) + ":"
                    ):
                        axis, MAGNIFICATION = (
                            float(line.split()[2][:-1]),
                            float(line.split()[4][:-1]),
                        )
                # transform 2D spike coordinates in projection
                tilt_X = T[0] * tilt_x + T[1] * tilt_y + T[4] + center_X
                tilt_Y = T[2] * tilt_x + T[3] * tilt_y + T[5] + center_Y

                # The true 2D coordiantes on raw micrographs
                tilt_X_true = tilt_X - min_micrograph_x
                tilt_Y_true = tilt_Y - min_micrograph_y

                # HF: re-center using translational shifts from sub-tomogram averaging
                fp = spa_euler_angles(
                    tilt,
                    -axis,
                    [norm0, norm1, norm2],
                    [
                        m00,
                        m01,
                        m02,
                        m03,
                        m04,
                        m05,
                        m06,
                        m07,
                        m08,
                        m09,
                        m10,
                        m11,
                        m12,
                        m13,
                        m14,
                        m15,
                    ],
                    cutOffset,
                )

                # add x y shifts from STA
                tilt_X_true += float(fp[3]) * int(parameters["extract_bin"])
                tilt_Y_true += float(fp[4]) * int(parameters["extract_bin"])
                fp[3] = 0.0
                fp[4] = 0.0

                # make them integers and store the errors in the columns of parfile
                tilt_X = int(math.floor(tilt_X_true))
                tilt_Y = int(math.floor(tilt_Y_true))

                tilt_X_err = 0 # (tilt_X_true - tilt_X) * float(parameters["scope_pixel"])
                tilt_Y_err = 0 # (tilt_Y_true - tilt_Y) * float(parameters["scope_pixel"])

                # check if particle completely inside micrograph (skip if not inside)
                if (
                    tilt_X - (cutboxsize / 2.0) < min_micrograph_x
                    or tilt_X - (cutboxsize / 2.0) + min_micrograph_x
                    >= max_micrograph_x
                    or tilt_Y - (cutboxsize / 2.0) < min_micrograph_y
                    or tilt_Y - (cutboxsize / 2.0) + min_micrograph_y
                    >= max_micrograph_y
                ):
                    logger.debug(
                        "Skipping particle outside image range: [%d,%d] x=(%d,%d), y=(%d,%d)."
                        % (
                            tilt_X - (cutboxsize / 2) + min_micrograph_x,
                            tilt_Y - (cutboxsize / 2) + min_micrograph_y,
                            min_micrograph_x,
                            max_micrograph_x,
                            min_micrograph_y,
                            max_micrograph_y,
                        )
                    )
                    tilt_image_counter += 1
                    continue
            
                # retrieve shifts from previous run
                x_correction = y_correction = 0
                if Path(refinement).exists() and ".par" in refinement:
                    local = input[input[:, 7] == film]
                    x_correction = local[image_counter - 1, 4] / pixel
                    y_correction = local[image_counter - 1, 5] / pixel

                    # undo in-plane rotation
                    # psi = -math.radians( local[image_counter-1, 1 ] )
                    # c, s = np.cos(psi), np.sin(psi)
                    # R = np.array(((c,-s),(s,c)))
                    # x = np.array(( x_correction, y_correction ))
                    # correction = np.matmul( R, x )
                    # x_correction, y_correction = correction[0], correction[1]

                    # print 'corrections for particle', film, image_counter-1, x_correction, y_correction
                """
                # check if too close to previously selected projections
                if len(allboxes) > 0:
                    # convert current position to np array
                    vector = np.array(allboxes, dtype="f")
                    # find all projections from this micrograph
                    vector = vector[vector[:, 2] == (tilt_image_counter - 1)][:, :2]
                    # calculate distance to all positions
                    if len(vector) > 0:
                        dmin = scipy.spatial.distance.cdist(
                            np.array(
                                [
                                    tilt_X - min_micrograph_x + x_correction,
                                    tilt_Y - min_micrograph_y + y_correction,
                                ],
                                ndmin=2,
                            ),
                            vector,
                        ).min()
                        if dmin < float(parameters["particle_rad"]) / float(
                            parameters["scope_pixel"]
                        ):
                            logger.warning(
                                "Skipping projection %s too close to already selected position (distance of %f pixels)"
                                % (tilt, dmin)
                            )
                            tilt_image_counter += 1
                            continue
                """

                allimodboxes.append(
                    "%2.0f\t%2.0f\t%2.0f\n"
                    % (
                        tilt_X  # - min_micrograph_x + cutboxsize / 2
                        + x_correction,
                        tilt_Y  # - min_micrograph_y # + cutboxsize / 2
                        + y_correction,
                        tilt_image_counter - 1,
                    )
                )

                # use defocus for this tilt

                df1 = defocus_per_tilt[tilt_image_counter - 1, 1]
                df2 = defocus_per_tilt[tilt_image_counter - 1, 2]
                angast = defocus_per_tilt[tilt_image_counter - 1, 3]
            
                if parameters["csp_ctf_handedness"]:
                    ctf_tilt_angle = angle * -1
                else:
                    # i.e. EMPIAR-10164
                    ctf_tilt_angle = angle

                # x-distance of tilt-axis to center of image
                # variable axis is from invert transform back to raw micrographs and is right handedness; positive angle - counterclockwise rotation
                tilt_axis_radians = math.radians(-axis)
                x_shift = tilt_series_alignment[tilt_image_counter - 1, -2]
                y_shift = tilt_series_alignment[tilt_image_counter - 1, -1]

                distance_to_axis = (
                    math.cos(tilt_axis_radians) * x_shift
                    - math.sin(tilt_axis_radians) * y_shift
                )
                tilt_based_height = (
                    distance_to_axis * math.tan(ctf_tilt_angle) * actual_pixel
                )

                defocus_offset = (
                    -tilt_based_height
                    + height_specimen * math.cos(ctf_tilt_angle)
                    - actual_pixel * (spike_Z - center_Z) * math.cos(ctf_tilt_angle)
                    + actual_pixel * (spike_X - center_X) * math.sin(ctf_tilt_angle)
                )
                # defocus_offset = DefocusOffsetFromCenter( transformed_3d_loc, [center_X, center_Y, center_Z], tilt, T, 300.0 ) * actual_pixel
                df1 += defocus_offset
                df2 += defocus_offset

                # print( "%d = "%(tilt_image_counter), defocus_offset )
                if not float(parameters["scope_cs"]) > 0:
                    df1 = 0
                    df2 = 0

                # this is the global particle index (from the autopick txt file)
                ptl_index = global_spike_counter
                dose = 0
                scan_order = int(scan_order_list[tilt_image_counter - 1])

                # use confidence column to store frame index
                confidence = 0

                ptl_CCX = tilt_image_counter

                # format parameter sequence and add to current .par file
                ppsi = ptheta = pphi = 0
                mag = float(parameters["scope_mag"])

                occ = 100
                sigma = 1
                score = random.uniform(0, 1) * 10
                # score = 0.5
                logp = change = 0

                # reset shifts and re-use parameters if we are re-centering boxes
                if Path(refinement).exists() and ".par" in refinement:
                    local = input[input[:, 7] == film]
                    fp[0] = local[image_counter - 1, 3]
                    fp[1] = local[image_counter - 1, 2]
                    fp[2] = local[image_counter - 1, 1]
                    fp[3] = fp[4] = 0
                    mag = local[image_counter - 1, 6]
                    film = local[image_counter - 1, 7]
                    df1 = local[image_counter - 1, 8]
                    df2 = local[image_counter - 1, 9]
                    angast = local[image_counter - 1, 10]
                    # occ = local[image_counter-1,11]
                    logp = local[image_counter - 1, 12]
                    sigma = local[image_counter - 1, 13]
                    score = local[image_counter - 1, 14]
                    change = local[image_counter - 1, 15]

                    tilt = local[image_counter - 1, 17]
                    axis = -local[image_counter - 1, 22]
                    ppsi = local[image_counter - 1, -3]
                    ptheta = local[image_counter - 1, -2]
                    pphi = local[image_counter - 1, -1]

                # AB - 0verride film number if we are processing each movie independently
                if parameters["csp_no_stacks"]:
                    film = 0

                if use_frames:
                    # if using frames

                    for idx_frame, frame_shifts in enumerate(
                        xf_frames[tilt_image_counter - 1]
                    ):
                        # store frame index in confidence column
                        confidence = idx_frame

                        frame_tilt_X_err = tilt_X_err - (
                            frame_shifts[4] * float(parameters["scope_pixel"])
                        )
                        frame_tilt_Y_err = tilt_Y_err - (
                            frame_shifts[5] * float(parameters["scope_pixel"])
                        )

                        allparxs[0].append(
                            frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE_WO_NO
                            % (
                                fp[0],
                                fp[1],
                                fp[2],
                                fp[3] * pixel,
                                fp[4] * pixel,
                                mag,
                                film,
                                df1,
                                df2,
                                angast,
                                occ,
                                logp,
                                sigma,
                                score,
                                change,
                                ptl_index,
                                tilt,
                                dose,
                                scan_order,
                                confidence,
                                ptl_CCX,
                                -axis,
                                norm0,
                                norm1,
                                norm2,
                                m00,
                                m01,
                                m02,
                                m03 * pixel, # 0.0,
                                m04,
                                m05,
                                m06,
                                m07 * pixel, # 0.0,
                                m08,
                                m09,
                                m10,
                                m11 * pixel, # 0.0,
                                m12,
                                m13,
                                frame_tilt_X_err,
                                frame_tilt_Y_err,
                                ppsi,
                                ptheta,
                                pphi,
                            )
                        )
                        allboxes.append(
                            [
                                tilt_X + x_correction,
                                tilt_Y + y_correction,
                                tilt_image_counter - 1,
                                idx_frame,
                            ]
                        )

                else:
                    # if NOT using frames
                    allparxs[0].append(
                        frealign_parfile.EXTENDED_NEW_PAR_STRING_TEMPLATE_WO_NO
                        % (
                            fp[0],
                            fp[1],
                            fp[2],
                            fp[3] * pixel,
                            fp[4] * pixel,
                            mag,
                            film,
                            df1,
                            df2,
                            angast,
                            occ,
                            logp,
                            sigma,
                            score,
                            change,
                            ptl_index,
                            tilt,
                            dose,
                            scan_order,
                            confidence,
                            ptl_CCX,
                            -axis,
                            norm0,
                            norm1,
                            norm2,
                            m00,
                            m01,
                            m02,
                            0.0, # m03 * pixel, # 0.0,
                            m04,
                            m05,
                            m06,
                            0.0, # m07 * pixel, # 0.0,
                            m08,
                            m09,
                            m10,
                            0.0, # m11 * pixel, # 0.0,
                            m12,
                            m13,
                            tilt_X_err,
                            tilt_Y_err,
                            ppsi,
                            ptheta,
                            pphi,
                        )
                    )
                    allboxes.append(
                        [
                            tilt_X + x_correction,
                            tilt_Y + y_correction,
                            tilt_image_counter - 1,
                        ]
                    )

                tilt_image_counter += 1
                image_counter += 1
            
            global_spike_counter += 1


    if len(allimodboxes) > 0:
        with open("%s_boxes.txt" % (name), "w") as f:
            f.writelines("%s\n" % item for item in allimodboxes)
    else:
        for f in glob.glob("%s_boxes.txt" % name):
            os.remove(f)

    if len(allboxes_3d) > 0:
        with open("csp/%s_boxes3d.txt" % (name), "w") as f:
            f.write(
                "%8s\t%8s\t%8s\t%8s\t%8s\t%8s\n"
                % ("PTLIDX", "X", "Y", "Z", "Score", "Keep_CSP")
            )
            f.writelines(
                "%8d\t%8.1f\t%8.1f\t%8.1f\t%8.2f\t%8s\n"
                % (idx, item[0], item[1], item[2], 0.0, "Yes")
                for idx, item in enumerate(allboxes_3d)
            )

    # convert coordinates to IMOD models
    if os.path.exists(name + "_boxes.txt"):
        com = "{0}/bin/point2model -input {1}_boxes.txt -output mod/{1}_boxes.mod -scat -circle 25".format(
            get_imod_path(), name
        )
        local_run.run_shell_command(com)
        os.remove(name + "_boxes.txt")

    if os.path.exists(name + "_ali_boxes.txt"):
        com = "{0}/bin/point2model -input {1}_ali_boxes.txt -output mod/{1}_ali_boxes.mod -scat -circle 25".format(
            get_imod_path(), name
        )
        local_run.run_shell_command(com)
        os.remove(name + "_ali_boxes.txt")

    return allboxes, allparxs

@timer.Timer(
    "get image particle index", text="Get image particles index took: {}", logger=logger.info
)
def get_image_particle_index(parameters, parfile, path="."):
    """
    Calculate index for each micrograph to quickly access to the particles that belong to it in the parfiles
    """
    micrographs = "{}.films".format(parameters["data_set"])
    micrograph_list = np.loadtxt(micrographs, dtype=str, ndmin=2)
    film_col = 7
    index = {}
    start = 0
    par_data = frealign_parfile.Parameters.from_file(parfile, toColumn=film_col).data[:, :8]
    for id in range(len(micrograph_list)):
        end = par_data[par_data[:, film_col] == id].shape[0]
        index.update({int(id): (start, start + end)})
        start += end 

    index_file = path + "/micrograph_particle.index"

    with open(index_file, 'w') as fp:
        json.dump(index, fp, indent=4,separators=(',', ': '))


def compute_global_weights(parfile: str, weights_file: str = "global_weight.txt"):

    FILM_COL = 8 - 1
    SCANORD_COL = 20 - 1
    OCC_COL = 12 - 1
    SCORE_COL = 15 - 1 

    import pandas as pd
    par_data = frealign_parfile.Parameters.from_file(parfile).data
    
    # only consider projections that have occ > 0
    par_data = par_data[par_data[:, OCC_COL] > 0.0]
    df = pd.DataFrame(data=par_data[:, [SCORE_COL, SCANORD_COL]], 
                      columns=["SCORE", "SCANORD"])
    
    sum_scores = df.groupby("SCANORD").sum()
    counts = df.groupby("SCANORD").size().reset_index(name='count')

    weights = []

    for scanord, row in sum_scores.iterrows():
        sum_score = row["SCORE"]

        while scanord >= len(weights):
            weights.append(-1.0)

        weights[int(scanord)] = sum_score
    
    for index, row in counts.iterrows():
        scanord = row["SCANORD"]
        count = row["count"]

        weights[int(scanord)] /= count

    with open(weights_file, "w") as f:
        f.write("\n".join(list(map(str, weights))))
                   

@timer.Timer(
    "get index of particle frames", text="Get index of particles frames took: {}", logger=logger.info
)
def get_particles_tilt_index(parfile, path="./"):
    """
    Calculate index for each particles to quickly access to the particle tilt series that belong to it in the parfile
    idea from alimanfoo/find_runs.py
    """

    index = {}
    par_data = frealign_parfile.Parameters.from_file(parfile).data
    
    if par_data.shape[1] > 45:
        ptl_index = 17
    else:
        ptl_index = 16
    
    ptlid = par_data[:, ptl_index].ravel()
    n = ptlid.shape[0]
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(ptlid[:-1], ptlid[1:], out=loc_run_start[1:])

    sections = np.nonzero(loc_run_start)
    index = np.append(sections, n)
    index = np.hstack((np.reshape(index[:-1], (-1, 1)), np.reshape(index[1:], (-1, 1))))
    index_file = path + "/particle_tilt.index"
    np.savetxt(index_file, index, fmt='%d')
    

def array2formatstring(a, format):
    return format % tuple(a)


