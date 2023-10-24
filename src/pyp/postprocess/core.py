import os
import random
import shutil
import subprocess
import numpy as np

from pyp.system import project_params
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import eman_load_command, get_relion_path, phenix_load_command, get_frealign_paths
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def fsigmoid(x, a, b):
    return 1.0 - 1.0 / (1.0 + np.exp(-a * (x - b)))


def get_rhref(mparameters, iteration, cutoff=0.143, factor=0.9):
    """
    Derive high resolution cutoff from current resolution estimate.
    """
    rhref = float(project_params.param(mparameters["refine_rhref"], iteration))
    if rhref > 0:
        return rhref
    elif rhref < 0:
        spread = 1.0 / rhref / 25.0
        return -1.0 / (1.0 / rhref + random.uniform(-spread, spread))
    else:
        fsc_file = "../maps/%s_r01_fsc.txt" % mparameters["refine_dataset"]
        if os.path.isfile(fsc_file) and iteration > 2:

            fsc = np.loadtxt(fsc_file, ndmin=2, dtype=float)

            # 1. Resolution limit based on fixed FSC cutoff
            # return factor / pyp_fsc.fsc_cutoff( fsc[:,[0,iteration-2]], cutoff )

            # 2. Resolution limit based on FSC value at previous cutoff

            # determine FSC at previous resolution cutoff
            res_file = "../maps/%s_r01_res.txt" % mparameters["refine_dataset"]
            if os.path.exists(res_file):
                prev_rhref = np.loadtxt(res_file, ndmin=2, dtype=float)[
                    iteration - 3, 1
                ]
            else:
                prev_rhref = 16.0

            import scipy.interpolate

            f = scipy.interpolate.interp1d(fsc[:, 0], fsc[:, iteration - 2])
            current = float(f(prev_rhref))

            # advance to the next resolution shell
            if current > cutoff:
                shells = int(current / cutoff)
                current_shell = np.argmin(np.abs(fsc[:, 0] - prev_rhref))
                return fsc[current_shell + shells, 0]
            else:
                return prev_rhref

            # 3. Resolution limit based on area below FSC
            area = 0
            for f in range(fsc.shape[0] - 1, 0 - 1, -1):
                area += fsc[f, iteration - 2] - fsc[f:, iteration - 2].mean()
                current = fsc[f, 0]
                if area > cutoff and current < rhref:
                    return current
                else:
                    return prev_rhref
        else:
            return 16


def measure_score(recfile, modelfile, resolution, scale, pixel, clip, flip):

    eman = eman_load_command()
    phenix = phenix_load_command()

    # TODO: remove eman2 dependency
    clipfile = recfile.replace(".mrc", "_clip.mrc")
    com = "{0}e2proc3d.py {1} {2} --clip={3} --scale={4} --origin=0,0,0 --apix={5}".format(
        eman, recfile, clipfile, clip, scale, pixel * scale
    )
    if flip:
        com += " --process xform.flip:axis=z"
    run_shell_command(com)

    com = "{0}phenix.mtriage {1} {2}".format(phenix, modelfile, clipfile)
    run_shell_command(com)

    masked_file = "fsc_model.masked.mtriage.log"
    unmasked_file = "fsc_model.unmasked.mtriage.log"
    import matplotlib.pyplot as plt

    plt.clf()
    fig = plt.figure(figsize=(16, 10))
    if os.path.exists(masked_file):
        masked = np.loadtxt(masked_file)
        plt.plot(masked[:, 0], masked[:, 1], label="Masked (%.2f)" % masked[:, 1].sum())
        atoms = masked[:, 1].sum()
    else:
        atoms = 0
    if os.path.exists(unmasked_file):
        unmasked = np.loadtxt(unmasked_file)
        plt.plot(
            unmasked[:, 0],
            unmasked[:, 1],
            label="Unmasked (%.2f)" % unmasked[:, 1].sum(),
        )
        plt.plot(unmasked[:, 0], 0.5 * np.ones_like(unmasked[:, 0]), "--", c="k")
        whole = unmasked[:, 1].sum()
    else:
        whole = 0
    plt.legend(fontsize=20)
    plt.ylim([-0.05, 1.0])
    plt.xlim([unmasked[0, 0], 1.0 / pixel / 2.0])
    plt.title(
        "FSC\n%s VS. %s" % (os.path.basename(recfile), os.path.basename(modelfile)),
        fontsize=20,
    )
    if os.path.exists("../maps"):
        plt.savefig("../maps/" + os.path.basename(recfile).replace(".mrc", "_pdb.png"))
    else:
        plt.savefig(os.path.basename(recfile).replace(".mrc", "_pdb.png"))

    try:
        os.remove("fsc_model.masked.mtriage.log")
        os.remove("fsc_model.unmasked.mtriage.log")
        os.remove("mask.ccp4")
    except:
        pass

    if os.path.exists(clipfile):
        os.remove(clipfile)

    return [whole, atoms]


def smooth_part_fsc(stats_file_name, plot_name):
    logger.info("Smoothing part FSC curves")
    stats = np.loadtxt(stats_file_name, comments=["C"])
    if len(stats) > 0:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        plt.plot(1.0 / stats[:, 1], stats[:, 4], label="Raw")

        window_len = min(40,stats.shape[0])
        window = "hanning"
        w = eval("np." + window + "(window_len)")

        for i in range(4, 7):
            x = stats[:, i]
            s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
            stats[:, i] = np.convolve(w / w.sum(), s, mode="same")[
                window_len - 1 : -window_len + 1
            ]
        if not stats_file_name.endswith("_raw"):
            shutil.copy2(stats_file_name, stats_file_name + "_raw")
        else:
            stats_file_name = stats_file_name.replace("_raw", "")
        np.savetxt(
            stats_file_name, stats, fmt="%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f%14.5f"
        )
        plt.plot(1.0 / stats[:, 1], stats[:, 4], label="Smooth")
        plt.plot(1.0 / stats[:, 1], np.zeros_like(stats[:, 4]), "--", c="k")
        plt.legend(fontsize=20)
        plt.xlim([0, 1.0 / stats[-1, 1]])
        plt.ylim([-0.25, 1.05])
        plt.title(os.path.basename(plot_name).split(".")[0])
        plt.savefig(plot_name)

def cistem_postprocess(args,output):
    """
            **   Welcome to SharpenMap   **

                Version : 1.00
                Compiled : Mar 27 2020
                    Mode : Interactive

    Input volume file name [T20S_r01_04.mrc]           :
    Output sharpened volume file name [output.mrc]     :
    Input mask file name [mask.mrc]                    :
    Input reconstruction statistics
    [statistics_r01.txt]                               :
    Use statistics [No]                                :
    Pixel size (A) [1.0]                               :
    Inner mask radius (A) [0.0]                        :
    Outer mask radius (A) [100.0]                      :
    Low-res B-Factor (A^2) [0.0]                       :
    High-res B-Factor (A^2) [0.0]                      :
    Low resolution limit for spectral flattening (A)
    [8.0]                                              :
    High resolution limit (A) [3.0]                    :
    Filter edge width (A) [20.0]                       :
    Part_SSNR scale factor [1.0]                       :
    Use 3D mask [No]                                   :
    Invert handedness [No]                             :
    
    """

    sharpenlogfile = "/dev/null"
    command = (
        "{0}/sharpen_map << eot >> {1} 2>&1\n".format(
            get_frealign_paths()["frealignx"], sharpenlogfile
        )
        + "{0}\n".format(args["sharpen_cistem_input_map"])
        + "{0}\n".format(output)
        + "{0}\n".format(args["sharpen_cistem_mask"])
        + "{0}\n".format(args["sharpen_cistem_statistics"])
        + ( "Yes\n" if args["sharpen_cistem_use_statistics"] else "No\n" )
        + "{0}\n".format(args["scope_pixel"] * args["extract_bin"])
        + "{0}\n".format(args["sharpen_cistem_inner_mask_radius"])
        + "{0}\n".format(args["sharpen_cistem_outer_mask_radius"])
        + "{0}\n".format(args["sharpen_cistem_low_res_bfactor"])
        + "{0}\n".format(args["sharpen_cistem_high_res_bfactor"])
        + "{0}\n".format(args["sharpen_cistem_low_res_flattening"])
        + "{0}\n".format(args["sharpen_cistem_high_res_limit"])
        + "{0}\n".format(args["sharpen_cistem_filter_edge_width"])
        + "{0}\n".format(args["sharpen_cistem_part_ssnr_scale"])
        + ( "Yes\n" if args["sharpen_cistem_use_mask"] else "No\n" )
        + ( "Yes\n" if args["sharpen_cistem_invert_handedness"] else "No\n" )
        + "eot\n"
    )

    run_shell_command(command, verbose=args['slurm_verbose'])

def relion_postprocess(args):

    # create sym links to files

    auxs = ["relion_half1_class001_unfil.mrc", "relion_half2_class001_unfil.mrc"]
    for aux in auxs:
        if os.path.exists(aux):
            os.remove(aux)
    os.symlink(args["sharpen_first_half"], "relion_half1_class001_unfil.mrc")
    os.symlink(args["sharpen_second_half"], "relion_half2_class001_unfil.mrc")

    # name = args["sharpen_first_half"].replace(".mrc", "")

    """
    +++ RELION: command line arguments (with defaults for optional ones between parantheses) +++
    ====== General options =====
                                    --i : Input name of half1, e.g. run_half1_class001_unfil.mrc
                                --i2 () : Input name of half2, (default replaces half1 from --i with half2)
                    --o (postprocess) : Output rootname
                        --angpix (-1) : Pixel size in Angstroms
                    --half_maps (false) : Write post-processed half maps for validation
                    --mtf_angpix (-1.) : Pixel size in the original micrographs/movies (in Angstroms)
                    --molweight (-1) : Molecular weight (in kDa) of ordered protein mass
    ====== Masking options =====
                    --auto_mask (false) : Perform automated masking, based on a density threshold
            --inimask_threshold (0.02) : Density at which to threshold the map for the initial seed mask
                --extend_inimask (3.) : Number of pixels to extend the initial seed mask
                --width_mask_edge (6.) : Width for the raised cosine soft mask edge (in pixels)
                            --mask () : Filename of a user-provided mask (1=protein, 0=solvent, all values in range [0,1])
                --force_mask (false) : Use the mask even when the masked resolution is worse than the unmasked resolution
    ====== Sharpening options =====
                            --mtf () : User-provided STAR-file with the MTF-curve of the detector
                    --auto_bfac (false) : Perform automated B-factor determination (Rosenthal and Henderson, 2003)
                --autob_lowres (10.) : Lowest resolution (in A) to include in fitting of the B-factor
                --autob_highres (0.) : Highest resolution (in A) to include in fitting of the B-factor
                    --adhoc_bfac (0.) : User-provided B-factor (in A^2) for map sharpening, e.g. -400
    ====== Filtering options =====
        --skip_fsc_weighting (false) : Do not use FSC-weighting (Rosenthal and Henderson, 2003) in the sharpening process
                        --low_pass (0) : Resolution (in Angstroms) at which to low-pass filter the final map (0: disable, negative: resolution at FSC=0.143)
    ====== Local-resolution options =====
                    --locres (false) : Perform local resolution estimation
                --locres_sampling (25.) : Sampling rate (in Angstroms) with which to sample the local-resolution map
                --locres_maskrad (-1) : Radius (in A) of spherical mask for local-resolution map (default = 0.5*sampling)
                --locres_edgwidth (-1) : Width of soft edge (in A) on masks for local-resolution map (default = sampling)
            --locres_randomize_at (25.) : Randomize phases from this resolution (in A)
                --locres_minres (50.) : Lowest local resolution allowed (in A)
    ====== Expert options =====
                    --ampl_corr (false) : Perform amplitude correlation and DPR, also re-normalize amplitudes for non-uniform angular distributions
            --randomize_at_fsc (0.8) : Randomize phases from the resolution where FSC drops below this value
                --randomize_at_A (-1) : Randomize phases from this resolution (in A) onwards (if positive)
                --filter_edge_width (2) : Width of the raised cosine on the low-pass filter edge (in resolution shells)
                            --verb (1) : Verbosity
                    --random_seed (0) : Seed for random number generator (negative value for truly random)
                            --version : Print RELION version and exit
    """

    general = f" --i {args['sharpen_first_half']} --i2 {args['sharpen_second_half']} --angpix {args['scope_pixel']} --molweight {args['particle_mw']}"
    masking = f" --inimask_threshold {args['sharpen_inimask_threshold']} --extend_inimask {args['sharpen_extend_inimask']} --width_mask_edge {args['sharpen_width_mask_edge']}"
    if args["sharpen_auto_mask"]:
        masking += " --auto_mask"
    if args["sharpen_force_mask"]:
        masking += " --force_mask"
    if "sharpen_mask" in args and args["sharpen_mask"] != None and os.path.exists(project_params.resolve_path(args["sharpen_mask"])):
        masking += f" --mask {project_params.resolve_path(args['sharpen_mask'])}"
    sharpening = ""
    if args["sharpen_auto_bfac"]:
        sharpening += f" --auto_bfac --autob_lowres {args['sharpen_autob_lowres']} -autob_highres {args['sharpen_autob_highres']}"
    if args["sharpen_adhoc_bfac"]:
        sharpening += f" --adhoc_bfac {args['sharpen_auto_bfac']}"
    filtering = f" --low_pass {args['sharpen_low_pass']}"
    if args["sharpen_skip_fsc_weighting"]:
        filtering += " --skip_fsc_weighting"
    local = ""
    if args["sharpen_locres"]:
        local += f" --locres_sampling {args['sharpen_locres_sampling']} --locres_maskrad {args['sharpen_locres_maskrad']} --locres_edgwidth {args['sharpen_locres_edgwidth']} --local_res_randomize_at {args['sharpen_locres_randomize_at']} --locres_minres {args['sharpen_locres_minres']}"
    expert = f" --randomize_at_fsc {args['sharpen_randomize_at_fsc']} --randomize_at_A {args['sharpen_randomize_at_A']} --filter_edge_width {args['sharpen_filter_edge_width']}"
    if args["sharpen_ampl_corr"]:
        expert += " --ampl_corr"

    com = "{0}/relion_postprocess".format( get_relion_path() ) + general + masking + sharpening + filtering + local + expert
    results, _ = run_shell_command(com)
    logger.info(results)

    star_file = "postprocess.star"

    # plot results
    post = open(star_file).read().split("data_")
    data = np.array(
        [
            line.split()
            for line in post[2]
            .split("_rlnFourierShellCorrelationUnmaskedMaps #4")[-1]
            .split("\n")
            if len(line) > 3
        ]
    )
    if len(data.shape) == 1:
        data = np.array(
            [
                line.split()
                for line in post[2]
                .split(
                    "_rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps #7"
                )[-1]
                .split("\n")
                if len(line) > 3
            ]
        )

    FinalResolution = [
        float(line.split()[1].replace("#", ""))
        for line in open(star_file)
        if "_rlnFinalResolution" in line
    ][0]
    BfactorUsedForSharpening = [
        float(line.split()[1].replace("#", ""))
        for line in open(star_file)
        if "_rlnBfactorUsedForSharpening" in line
    ][0]

    import matplotlib.pyplot as plt
    plt.plot(data[:, 1].astype("f"), data[:, 3].astype("f"), label="Corrected")
    if len(data.shape) > 1:
        plt.plot(
            data[:, 1].astype("f"), data[:, 4].astype("f"), label="Unmasked Maps"
        )
        plt.plot(
            data[:, 1].astype("f"), data[:, 5].astype("f"), label="Masked Maps"
        )
        plt.plot(
            data[:, 1].astype("f"),
            data[:, 6].astype("f"),
            label="Corrected PR Masked Maps",
        )
    # plt.ylim( [-.05,1.05] )
    plt.plot(data[:, 1].astype("f"), 0.143 * np.ones(data[:, 1].shape), "k:")
    plt.plot(data[:, 1].astype("f"), 0.5 * np.ones(data[:, 1].shape), "k:")
    plt.ylim((-0.1, 1.05))
    plt.xlim((data[:, 1].astype("f")[0], data[:, 1].astype("f")[-1]))
    plt.legend(prop={"size": 10})
    plt.title(
        "%s\nFinal Resolution = %.2f, Bfactor = %.2f"
        % (
            first_half.replace("_crop.mrc", ""),
            FinalResolution,
            BfactorUsedForSharpening,
        )
    )
    plt.savefig(first_half.replace(".mrc", "_relion.png"))

    # clean up
    shutil.move("postprocess.star", first_half.replace(".mrc", ".star"))
    shutil.move(
        "postprocess_masked.mrc", first_half.replace(".mrc", "_postprocess.mrc")
    )
    list = [
        "relion_half1_class001_unfil.mrc",
        "relion_half2_class001_unfil.mrc",
        "postprocess.mrc",
        "postprocess_automask.mrc",
        "postprocess_fsc.xml",
    ]
    if not args.keep:
        [os.remove(f) for f in list if os.path.exists(f)]
