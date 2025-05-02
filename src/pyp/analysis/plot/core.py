import glob
import math
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.spatial import distance

import pyp.analysis.scores
import pyp.inout.image as imageio
import pyp.inout.metadata as metaio
from pyp.inout.image import img2webp
from pyp.inout.metadata import pyp_metadata, cistem_star_file
from pyp.analysis.image import contrast_stretch
from pyp.system import project_params
from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system.utils import get_imod_path, check_env
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def contact_sheet(Y, cols=25, rescale=True, order: list = None):
    if rescale:
        X = Y
    else:
        X = 255.0 * (Y - Y.min()) / (Y.max() - Y.min())
    count, m, n = X.shape
    m += 2
    n += 2
    mm = int(math.ceil(math.sqrt(count)))
    nn = min(cols, count)
    mm = int(math.ceil(count / float(nn)))
    M = np.ones((mm * m, nn * n)) * X.mean()
    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            class_id = image_id if order is None else order[image_id]
            data = np.flipud(X[class_id, :, :])
            if rescale:
                rescaled = (
                    255.0 * (data - data.min()) / (data.max() - data.min())
                ).astype(np.uint8)
            else:
                rescaled = data.astype(np.uint8)
            M[sliceM + 1 : sliceM + m - 1, sliceN + 1 : sliceN + n - 1] = rescaled
            image_id += 1
    M = np.flipud(M)
    return M


def guinier_plot(weights, filename="weights.png", pixel_size=1):

    # accumulate weights
    frames = weights.shape[0]
    points = weights.shape[-1]
    cummulative = np.zeros([frames, points])
    areas = np.empty([frames])
    for i in range(frames):
        if i > 0:
            if weights.ndim > 2:
                cummulative[i] = cummulative[i - 1, :] + weights[i, :points, 0]
            else:
                cummulative[i] = cummulative[i - 1, :] + weights[i, :]
        else:
            if weights.ndim > 2:
                cummulative[i, :] = weights[i, :points, 0]
            else:
                cummulative[i] = weights[i, :]
        if weights.ndim > 2:
            areas[i] = weights[i, :points, 0].sum()
        else:
            areas[i] = weights[i, :].sum()
    ranked = np.argsort(areas)

    plt.clf()
    if "png" in filename:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=(10, 10))
    freq = 1.0 * np.arange(points) / points / pixel_size / 2
    for i in range(weights.shape[0]):
        # print i, ranked[i], 256*(i+1)/frames
        if ranked[i] == 0:
            plt.fill_between(
                freq,
                np.zeros(cummulative[ranked[i], :].shape),
                cummulative[0, :],
                facecolor=plt.cm.coolwarm(1.0 * (i + 1) / frames),
                linewidth=0.2,
                edgecolor="white",
            )
        elif ranked[i] + 1 < weights.shape[0]:
            plt.fill_between(
                freq,
                cummulative[ranked[i] - 1, :],
                cummulative[ranked[i], :],
                facecolor=plt.cm.coolwarm(1.0 * (i + 1) / frames),
                linewidth=0.2,
                edgecolor="white",
            )
        else:
            plt.fill_between(
                freq,
                cummulative[ranked[i] - 1, :],
                np.ones(cummulative[ranked[i], :].shape),
                facecolor=plt.cm.coolwarm(1.0 * (i + 1) / frames),
                linewidth=0.2,
                edgecolor="white",
            )
    plt.xlim([freq[0], freq[-1]])
    plt.ylim([0, 1])
    a = plt.gca()
    a.set_frame_on(False)
    # a.set_xticks([]); a.set_yticks([])
    # plt.axis('off')
    plt.xlabel("Frequency (1/$\mathregular{\AA}$)", fontsize=20, fontweight='semibold', labelpad=10)
    plt.ylabel("Cumulative weights", fontsize=20, fontweight='semibold', labelpad=10)
    
    ticks = [(0, "0"), (0.1, "1/10"), (0.2, "1/5"), (0.3, "1/3.0"), (0.5, "1/2.0"), (1.0 / (pixel_size * 2.0), f"1/{pixel_size * 2.0}") ]
    ticks = [ _ for _ in ticks if _[0] <= 1.0 / (pixel_size * 2.0) ]
    
    xticks = [_[0] for _ in ticks]
    xlabels = [_[1] for _ in ticks]
        
    plt.xticks(xticks, xlabels)

    # plt.locator_params(axis='x',nbins=5)
    plt.locator_params(axis="y", nbins=2)
    plt.tick_params(labelsize=15)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    # plt.savefig( filename, bbox_inches='tight', pad_inches=0 )


def plot_angular_trajectory(angles, output_name="", noisy=np.empty([0]), savefig=True):
    # plot trajectory
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="white")

    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D

    # Create a sphere
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:20j, 0.0 : 2.0 * pi : 20j]
    x = sin(phi) * cos(theta)
    y = sin(phi) * sin(theta)
    z = cos(phi)

    data = np.radians(angles)

    xx = sin(data[:, 0]) * cos(data[:, 1])
    yy = sin(data[:, 0]) * sin(data[:, 1])
    zz = cos(data[:, 0])

    # Set colours and render
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    from matplotlib.colors import LightSource

    ls = LightSource(270, 45)
    rgb = ls.shade(z, cmap=cm.cool, vert_exag=0.1, blend_mode="soft")

    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        color="c",
        alpha=0.15,
        linewidth=1,
        facecolors=rgb,
        shade=False,
    )

    if len(noisy) > 0:
        data = np.radians(noisy)

        xxi = sin(data[:, 0]) * cos(data[:, 1])
        yyi = sin(data[:, 0]) * sin(data[:, 1])
        zzi = cos(data[:, 0])

        ax.scatter(xxi, yyi, zzi, color="b", marker="o", s=30, label="noisy")

    ax.scatter(xx, yy, zz, color="r", marker="s", s=30, label="regularized")

    xv = angles[:, 0].mean()
    yv = angles[:, 1].mean()

    # print xv, yv

    ax.view_init(90 - xv, yv)
    ax.axis("off")
    limit = 0.6
    # limit = 0.15
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_aspect("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()
    if savefig:
        if len(output_name) > 0:
            plt.savefig(output_name)
        else:
            pass
            # plt.savefig('{}_rot_xf.png'.format(name))
    plt.close()


def plot_ccc(name):
    # plot cumulative correlation values
    import matplotlib.pyplot as plt

    ccc = np.loadtxt("{}.ccc".format(name), dtype=float)
    c = np.linspace(0, ccc.size - 1, ccc.size)
    fig, ax = plt.subplots(figsize=(2.56, 1.28), dpi=100)
    ax.plot(c, ccc, "r.-", label="cc-per-frame")
    if os.path.exists(name + ".ddd"):
        ddd = np.loadtxt("{}.ddd".format(name), dtype=float)
        d = (
            np.linspace(0, ddd.size - 1, ddd.size)
            / float(ddd.size)
            * float(ccc.size - 1)
        )
        ax.plot(d, ddd, "b.-", label="cc-per-iteration")
    ax.set_xlim((0, ccc.size - 1))
    ax.set_xlim((0, ccc.size - 1))
    ax.legend(fontsize=6)
    plt.tick_params(labelsize=8)
    plt.savefig("{}_ccc.png".format(name))
    plt.close()


def plot_dataset(parameters, current_count=-9, work_dir="", auto="."):

    inputlist = [
        os.path.splitext(f)[0].replace("ctf/", "") for f in glob.glob("ctf/*.ctf")
    ]
    indexes = np.argsort(
        [os.path.splitext(f)[0].replace("ctf/", "")[-9:] for f in inputlist]
    )
    inputlist = [inputlist[i] for i in indexes]

    try:
        data_set = parameters["data_set"]
    except KeyError:
        data_set = None

    if len(inputlist) >= current_count + 10:

        current_count = len(inputlist) // 10 * 10

        metaio.compileDatabase(inputlist, "{}_dbase.txt".format(data_set))

        if work_dir:
            os.chdir(work_dir)

        # read as dictionary
        # with open( '{}_dbase.txt'.format( parameters['data_set'] ) ) as data:
        #    reader = csv.DictReader( data, delimiter='\t' )

        data = np.loadtxt(
            auto + "/{}_dbase.txt".format(data_set), comments="Micrograph", dtype="str",
        )

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="darkgrid")

        # DF = 12
        # CC = 13
        # CCC = 14
        # COUNTS
        # DF1
        # DF2
        # CCC
        # X
        # Y
        # Z

        # First group of plots
        fig = plt.figure(figsize=(8, 1.5))

        my_cmap = sns.cubehelix_palette(dark=0.3, light=0.8, as_cmap=True)

        # CC histogram
        ax = plt.subplot(1, 4, 1)
        ax.hist(
            data[:, 19].astype("f"), bins=100, density=True, alpha=0.5, color="#0070FF"
        )
        ax.set_title("CTF Fit Distribution", fontsize=4)
        ax.tick_params(labelsize=4)

        # Defocus histogram
        ax = plt.subplot(1, 4, 2)
        # ax.hist(data[:,12].astype('f')*1e-4, bins=100, density=True, alpha=0.5, color="#F070FF")
        ax.hist(
            data[:, 14].astype("f"),
            bins=data.shape[0],
            density=True,
            alpha=0.5,
            color="#F070FF",
        )
        ax.set_title("Estimated Resolution Distribution (A)", fontsize=3)
        resolution_values = data[:, 14].astype("f")
        max_res = 1.5 * np.median(resolution_values)
        ax.set_xlim([2.5, max_res])
        ax.tick_params(labelsize=4)

        seq = np.arange(data.shape[0]) / (1.0 * data.shape[0]) * 256

        # DF1 vs DF2
        ax = plt.subplot(1, 4, 3)
        # sns.jointplot( data[:,16].astype('f')*1e-4, data[:,17].astype('f')*1e-4 )
        ax.scatter(
            data[:, 16].astype("f") * 1e-4,
            data[:, 17].astype("f") * 1e-4,
            c=seq,
            cmap="cool",
            s=2,
            alpha=0.75,
        )
        # ax.hist(data[:,12].astype('f'), bins=50, density=True, alpha=0.5, color="#0070FF")
        ax.set_title("DF1 vs. DF2", fontsize=4)
        ax.tick_params(labelsize=4)
        ax.axis("equal")

        # Astigmatism
        ax = plt.subplot(1, 4, 4, projection="polar")
        theta = np.deg2rad(data[:, 18].astype("f"))
        rho = data[:, 23].astype("f")
        ax.scatter(theta, rho, s=2, c=seq, cmap="cool")
        ax.set_xlabel("Astigmatism", fontsize=4)
        ax.tick_params(labelsize=4)
        ax.set_ylim([np.median(rho), 2 * np.median(rho)])
        ax.set_yticks([np.median(rho), 2 * np.median(rho)])
        plt.savefig(
            "{}/.{}_stats.png".format(auto, data_set), bbox_inches="tight", dpi=300,
        )
        # plt.savefig( auto + '/.' + parameters['data_set'] + '_stats.pdf', bbox_inches='tight', dpi=500 )

        # Time dependent quantities
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=[8, 5], sharex=True)

        ax[0].set_title(
            "{} ({} micrographs, {} particles )".format(
                data_set, str(len(inputlist)), str(int(data[:, 26].astype("f").sum()))
            ),
            fontsize=6,
        )

        # plot mean defocus
        defocus_values = data[:, 12].astype("f") * 1e-4
        sns.swarmplot(x=np.arange(data.shape[0]), y=defocus_values, ax=ax[0], size=2)
        # ax[0].tick_params(labelbottom=False)
        ax[0].set_ylabel("Defocus (microns)", fontsize=4)
        ax[0].tick_params(labelsize=4)
        ax[0].plot(np.sort(defocus_values), c="gray", lw=0.5)

        # estimated resolution
        resolution_values = data[:, 14].astype("f")
        sns.swarmplot(x=np.arange(data.shape[0]), y=resolution_values, ax=ax[1], size=2)
        # ax[1].tick_params(labelbottom=False)
        ax[1].set_ylabel("Estimated Resolution (A)", fontsize=4)
        ax[1].tick_params(labelsize=4)
        max_res = 1.5 * np.median(resolution_values)
        ax[1].set_ylim([2.5, max_res])
        ax[1].plot(np.sort(resolution_values), c="gray", lw=0.5)

        # Average motion
        median = np.median(data[:, -2].astype("f"))
        motion_values = data[:, -2].astype("f")
        sns.swarmplot(x=np.arange(data.shape[0]), y=motion_values, ax=ax[2], size=2)
        # ax[2].tick_params(labelbottom=False)
        ax[2].set_ylabel("Average Motion (pixels)", fontsize=4)
        ax[2].tick_params(labelsize=4)
        ax[2].set_ylim([0, 5 * median])
        ax[2].plot(np.sort(motion_values), c="gray", lw=0.5)

        # number of particles
        sns.swarmplot(
            x=np.arange(data.shape[0]), y=data[:, 26].astype("f"), ax=ax[3], size=2
        )
        # ax[3].tick_params(labelbottom=False)
        ax[3].set_ylabel("Number of Particles", fontsize=4)
        ax[3].tick_params(labelsize=4)
        ax[3].plot(np.sort(data[:, 26].astype("f")), c="gray", lw=0.5)

        interval = max(1, len(inputlist) / 12)
        indexes = np.flip(np.arange(len(inputlist) - 1, 0, -interval), axis=0).astype(
            "int"
        )
        plt.xticks(indexes, sorted([f[-9:] for f in data[indexes, 0]]), rotation=25)
        plt.savefig(
            "{}/.{}_time.png".format(auto, data_set), bbox_inches="tight", dpi=300,
        )
        # plt.savefig( auto + '/.' + parameters['data_set'] + '_time.pdf', bbox_inches='tight', dpi=500 )
        plt.close("all")

        com = "montage {}/.{}_stats.png {}/.{}_time.png -mode concatenate -tile 1x {}/{}.webp".format(
            auto, data_set, auto, data_set, auto, data_set
        )
        run_shell_command(com, verbose=parameters["slurm_verbose"])
        os.remove("{}/.{}_stats.png".format(auto, data_set))
        os.remove("{}/.{}_time.png".format(auto, data_set))

        # if len(inputlist) % 250 == 0:
        #    user_comm( parameters['data_set'] + ' (' + str(len(inputlist)) + ')',  auto + '/' + parameters['data_set'] + '.png' )


def plot_trajectory(name, output_name="", savefig=True):

    isfile = isinstance(name, str)

    # plot trajectory
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import cm
    from matplotlib.ticker import FuncFormatter, LinearLocator

    sns.set(style="darkgrid")

    if isfile:
        xf = np.loadtxt("{}.xf".format(name), dtype=float, ndmin=2)
    else:
        xf = np.copy(name)
    fig, ax = plt.subplots(figsize=(30, 30), dpi=800)
    ax.plot(xf[:, 4], xf[:, 5], "k-")
    if isfile and os.path.exists(name + ".prexgraw"):
        xfraw = np.loadtxt("{}.prexgraw".format(name), dtype=float)
        ax.plot(xfraw[:, 4], xfraw[:, 5], "b.")
    c = np.linspace(0, xf.shape[0] - 1, xf.shape[0])
    if isfile and "_P" in name and "_frames" in name:
        ax.scatter(xf[:, 4], xf[:, 5], c=c, cmap=cm.winter)
    else:
        ax.scatter(xf[:, 4], xf[:, 5], c=c, cmap=cm.autumn)
    ax.annotate(
        "{}".format(0),
        xy=(xf[0, 4], xf[0, 5]),
        xytext=(-5, 5),
        ha="right",
        textcoords="offset points",
        size=8,
    )
    ax.annotate(
        "{}".format(xf.shape[0] // 2),
        xy=(xf[xf.shape[0] // 2, 4], xf[xf.shape[0] // 2, 5]),
        xytext=(-5, 5),
        ha="right",
        textcoords="offset points",
        size=8,
    )
    ax.annotate(
        "{}".format(xf.shape[0] - 1),
        xy=(xf[-1, 4], xf[-1, 5]),
        xytext=(-5, 5),
        ha="right",
        textcoords="offset points",
        size=8,
    )
    x = xf[:, 4]
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: ("%.1f") % x))
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    y = xf[:, 5]
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: ("%.1f") % y))
    ax.yaxis.set_major_locator(LinearLocator(numticks=4))
    ax.axis("equal")
    plt.tick_params(labelsize=8)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    # plt.show()
    if savefig:
        if len(output_name) > 0:
            plt.savefig(output_name)
        else:
            plt.savefig("{}_xf.png".format(name))
    plt.close()


def plot_trajectory_raw(xf, output_name="", noisy=np.empty([0])):

    # plot trajectory
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import cm
    from matplotlib.ticker import FuncFormatter, LinearLocator

    sns.set(style="darkgrid")

    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)

    if len(noisy):
        ax.plot(noisy[:, -2], noisy[:, -1], "b.", label="noisy")
    ax.plot(xf[:, -2], xf[:, -1], "k-", label="regularized")
    c = np.linspace(0, xf.shape[0] - 1, xf.shape[0])
    ax.scatter(xf[:, -2], xf[:, -1], c=c, cmap=cm.autumn)
    ax.annotate(
        "{}".format(0),
        xy=(xf[0, -2], xf[0, -1]),
        xytext=(-5, 5),
        ha="right",
        textcoords="offset points",
        size=8,
    )
    ax.annotate(
        "{}".format(xf.shape[0] / 2),
        xy=(xf[xf.shape[0] // 2, -2], xf[xf.shape[0] // 2, -1]),
        xytext=(-5, 5),
        ha="right",
        textcoords="offset points",
        size=8,
    )
    ax.annotate(
        "{}".format(xf.shape[0] - 1),
        xy=(xf[-1, -2], xf[-1, -1]),
        xytext=(-5, 5),
        ha="right",
        textcoords="offset points",
        size=8,
    )
    x = xf[:, -2]
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: ("%.1f") % x))
    ax.xaxis.set_major_locator(LinearLocator(numticks=4))
    y = xf[:, -1]
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: ("%.1f") % y))
    ax.yaxis.set_major_locator(LinearLocator(numticks=4))
    ax.axis("equal")
    plt.tick_params(labelsize=8)
    if len(noisy):
        plt.legend()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    if len(output_name) > 0:
        plt.savefig(output_name)
    else:
        pass
        # plt.savefig('{}_xf.png'.format(name))
    plt.close()


def generate_plots(
    pardata, output_name, angles=25, defocuses=25, scores=False, is_tomo=False, dump=False, tilt_min=0, tilt_max=0,
):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    (
        angular_group,
        defocus_group,
    ) = pyp.analysis.scores.assign_angular_defocus_groups(
        pardata, angles, defocuses
    )
    input = pardata

    par_obj = cistem_star_file.Parameters()
    field = par_obj.get_index_of_column(cistem_star_file.SCORE)

    ptlindex = par_obj.get_index_of_column(cistem_star_file.PIND)
    film_id = par_obj.get_index_of_column(cistem_star_file.IMAGE_IS_ACTIVE)
    occ_col = par_obj.get_index_of_column(cistem_star_file.OCCUPANCY)
    tind_col = par_obj.get_index_of_column(cistem_star_file.TIND)
    df1_col = par_obj.get_index_of_column(cistem_star_file.DEFOCUS_1)
    theta_col = par_obj.get_index_of_column(cistem_star_file.THETA)
    lgp_col = par_obj.get_index_of_column(cistem_star_file.LOGP)
    sgm_col = par_obj.get_index_of_column(cistem_star_file.SIGMA)

    defocus_values = np.zeros(defocuses)
    for f in range(defocuses):
        if input.shape[0] > 0:
            defocus_values[f] = np.where(
                np.logical_and(defocus_group == f, np.isfinite(input[:, field])),
                input[:, df1_col],
                0,
            ).max()
        else:
            defocus_values[f] = 0

    angular_values = np.zeros(angles)
    for g in range(angles):
        if input.shape[0] > 0:
            angular_values[g] = np.where(
                np.logical_and(angular_group == g, np.isfinite(input[:, field])),
                input[:, df1_col],
                0,
            ).max()
        else:
            angular_values[g] = 0

    # defocus-orientation PR plots
    plot = np.empty([angles, defocuses])
    count = np.empty(plot.shape)
    thresholds = np.zeros(plot.shape)
    summation = np.empty(plot.shape)
    for g in range(angles):
        for f in range(defocuses):
            
            cluster = np.logical_and(
                np.logical_and(angular_group == g, defocus_group == f),
                np.isfinite(input[:, field]),
            )
            count[g, f] = np.where(cluster, 1, 0).sum()
            summation[g, f] = np.where(cluster, input[:, field], 0).sum()
            if count[g, f] > 0:
                plot[g, f] = summation[g, f] / count[g, f]
            else:
                # plot[g, f] = np.nan
                plot[g, f] = 0

            # plot scores histogram and produce sorted stack
            if dump:

                fig, ax = plt.subplots(figsize=(7, 5))
                prs = np.extract(cluster == 1, input[:, field])
                if prs.size > 1:

                    # assume two gaussian mixture model and derive optimal threshold for separation automatically
                    from sklearn import mixture

                    gmix = mixture.GMM(n_components=2, covariance_type="full")
                    gmix.fit(prs.reshape([prs.size, 1]))
                    threshold = np.array(gmix.means_).mean()
                    cutoff = 1.0 - 1.0 * np.argmin(np.fabs(prs - threshold)) / prs.size
                    # print 'Bi-modal distributions detected with means: {0}, {1}'.format( gmix.means_[0][0], gmix.means_[1][0] )
                    # print 'Optimal threshold for bimodal separation is {0}, cutoff = {1}\n'.format( threshold, cutoff )
                    thresholds[g, f] = threshold

                    # ax.hist( np.array( prs, dtype=float), angles, normed=1, facecolor='blue', alpha=0.75)
                    ax.hist(
                        np.array(prs, dtype=float),
                        angles,
                        density=1,
                        facecolor="blue",
                        alpha=0.75,
                    )
                    plt.title(
                        "Orientation group %s, Defocus group %s (T=%0.2f)"
                        % (g, f, threshold)
                    )
                    # ax[0].set_yticks( [] )
                    # plt.subplots_adjust(left=0.02, right=.98, top=0.9, bottom=0.1)
                    plt.savefig(
                        "../maps/%s_V%04d_D%04d_hist.png"
                        % (output_name, g, f)
                    )

                    # create fmatch stack
                    fmatch_stack = "/scratch/{0}_match.mrc".format(output_name)
                    # fmatch_stack = '../maps/{0}_match.mrc'.format( output_name )
                    fmatch_stack = "../maps/{0}_match_unsorted.mrc".format(output_name)

                    if os.path.exists(fmatch_stack):
                        A = np.transpose(
                            np.vstack(
                                (
                                    np.extract(cluster == 1, input[:, 0]),
                                    np.extract(cluster == 1, input[:, field]),
                                )
                            )
                        )
                        if A.ndim == 1:
                            A = np.reshape(A, (-1, 2))
                        B = A[np.argsort(-A[:, -1])]
                        sorted_indexes = (B[:, 0] - 1).tolist()  # 0-based indexes
                        fmatch_stack_group = "../maps/%s_V%04d_D%04d_match.mrc" % (
                            output_name,
                            g,
                            f,
                        )
                        imageio.mrc.extract_slices(
                            fmatch_stack, sorted_indexes, fmatch_stack_group
                        )
                    else:
                        logger.info("File not found %s", fmatch_stack)

    if dump:
        fig = plt.figure(figsize=(5, 5))
        cax = plt.imshow(thresholds, interpolation="nearest", cmap=cm.jet)
        plt.title("Thresholds per orientation\n and defocus group")
        plt.xlabel("Defocus Group")
        plt.ylabel("Orientation Group")
        plt.colorbar(cax, ticks=[np.nanmin(thresholds), np.nanmax(thresholds)])
        plt.savefig("../maps/%s_thresholds.png" % output_name)
        np.savetxt(
            "../maps/%s_thresholds.txt" % output_name,
            thresholds,
            fmt="%10.5f",
        )

    if scores:
        # number of particles in this class (Occ>50%)
        # total = np.logical_and( input[:,11]>50, np.isfinite(input[:,field]) ).sum()
        total = int(np.where(np.isfinite(input[:, field]), input[:, occ_col], 0).sum() / 100)
    else:
        total = count.sum()

    fig = plt.figure(figsize=(8, 4))
    fig.subplots_adjust(left=0.075, right=0.925, top=0.95, bottom=0.05)
    fig.suptitle(
        "%s\n(%i of %i)"
        % (output_name, total, angular_group.shape[0]),
        fontsize=12,
        fontweight="bold",
    )

    metadata = {}
    metadata["particles_total"] = angular_group.shape[0]
    metadata["particles_used"] = total

    from mpl_toolkits.axes_grid1 import AxesGrid

    grid = AxesGrid(
        fig,
        111,  # similar to subplot(122)
        nrows_ncols=(1, 2),
        axes_pad=1,
        label_mode="all",
        cbar_location="right",
        cbar_mode="each",
        cbar_size="7%",
        cbar_pad="2%",
    )
    im = grid[0].imshow(count, interpolation="nearest", cmap=cm.jet)
    grid.cbar_axes[0].colorbar(im, ticks=[np.nanmin(count), np.nanmax(count)])

    grid[0].set_title("Histogram")
    grid[0].set_xlabel("Defocus (%d, %d)" % (defocus_values[0], defocus_values[-1]))
    grid[0].set_ylabel("Orientations")

    im = grid[1].imshow(plot, interpolation="nearest", cmap=cm.jet)
    grid.cbar_axes[1].colorbar(im, ticks=[np.nanmin(plot), np.nanmax(plot)])
    grid[1].set_title("Average Phase Residual")
    grid[1].set_xlabel("Defocus (%d, %d)" % (defocus_values[0], defocus_values[-1]))
    grid[1].set_ylabel("Orientations")

    plt.savefig("{}_prs.png".format(output_name))

    angular_group = np.floor(np.mod(input[:, theta_col], 180) * angles / 180)
    if input.shape[0] > 0:
        mind, maxd = (
            int(math.floor(input[:, df1_col].min())),
            int(math.ceil(input[:, df1_col].max())),
        )
    else:
        mind = maxd = 0
    if maxd == mind:
        defocus_group = np.zeros(angular_group.shape)
    else:
        defocus_group = np.round((input[:, df1_col] - mind) / (maxd - mind) * (defocuses - 1))

    fig, ax = plt.subplots(2, 3, figsize=(8, 4))
    rot_n, rot_bins, rot_patches = ax[0, 0].hist(
        angular_group, angles, density=1, facecolor="green", alpha=0.75
    )
    ax[0, 0].set_title("Orientations")
    ax[0, 0].set_yticks([])
    def_n, def_bins, def_patches = ax[0, 1].hist(
        defocus_group, defocuses, density=1, facecolor="red", alpha=0.75
    )
    ax[0, 1].set_title("Defocus")
    ax[0, 1].set_yticks([])

    # used = np.logical_and( input[:,occ_col]>50, np.isfinite(input[:,field]) )
    prs = sorted(input[:, field])

    scores_n, scores_bins, scores_patches = ax[0, 2].hist(
        np.array(prs, dtype=float), angles, density=1, facecolor="blue", alpha=0.75
    )
    phase_residual = 0
    if len(prs) > 0:
        phase_residual = np.array(prs, dtype=float).min()
        ax[0, 2].set_title("Phase Residuals (%.2f)" % phase_residual )
    else:
        ax[0, 2].set_title("Phase Residuals")
    ax[0, 2].set_yticks([])
    metadata["phase_residual"] = phase_residual

    # particle score plot for tomo
    try:
        if is_tomo:
            used_mask = np.logical_and(
                input[:, occ_col] > 50, 
                np.logical_and(
                    np.abs(input[:, tind_col]) <= tilt_max, 
                    np.abs(input[:, tind_col]) >= tilt_min
                    )
                )
            
            used_sub_data = pd.DataFrame(input[used_mask], columns=cistem_star_file.Parameters.HEADER_STRS)
            
            mean_score = used_sub_data.groupby(["IMAGE_IS_ACTIVE","PIND"])["SCORE"].mean()

            histogram_particle_tomo(mean_score, threshold=0, tiltseries=output_name, save_path="../maps")
    except:
        logger.info("Unable to produce particle score plot, too few particles maybe?")
        pass

    occ_n, occ_bins, occ_patches = ax[1, 0].hist(
        input[:, occ_col], angles, density=1, facecolor="yellow", alpha=0.75
    )
    occupancies = input[:, occ_col].mean()
    metadata["occ"] = occupancies
    ax[1, 0].set_xlabel("Occupancies = %.2f" % occupancies )
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xticks([])

    logp_n, logp_bins, logp_patches = ax[1, 1].hist(
        input[:, lgp_col], angles, density=1, facecolor="cyan", alpha=0.75
    )
    logp = input[:, lgp_col].mean()
    metadata["logp"] = logp
    ax[1, 1].set_xlabel("LogP = %.2f" % logp)
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xticks([])

    sigma_n, sigma_bins, sigma_patches = ax[1, 2].hist(
        input[:, sgm_col], angles, density=1, facecolor="magenta", alpha=0.75
    )
    sigma = input[:, sgm_col].mean()
    metadata["sigma"] = sigma
    ax[1, 2].set_xlabel("Sigma = %.2f" % sigma)
    ax[1, 2].set_yticks([])
    ax[1, 2].set_xticks([])

    plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
    plt.savefig("%s_hist.png" % output_name)

    # combine the two plos
    command = "montage {0}_prs.png {0}_hist.png -tile 1x2 -geometry +0+0 {0}_prs.png".format(
        output_name
    )
    run_shell_command(command, verbose=False)

    os.remove("%s_hist.png" % output_name)

    # outputs: counts, plot, orientations, defocus, scores, occ, logp and sigma (n, bins)
    output = {}
    output["def_rot_histogram"] = count.tolist()
    output["def_rot_scores"] = plot.tolist()

    output["rot_hist"] = {}
    output["rot_hist"]["n"] = rot_n.tolist()
    output["rot_hist"]["bins"] = rot_bins.tolist()
    output["def_hist"] = {}
    output["def_hist"]["n"] = def_n.tolist()
    output["def_hist"]["bins"] = def_bins.tolist()
    output["scores_hist"] = {}
    output["scores_hist"]["n"] = scores_n.tolist()
    output["scores_hist"]["bins"] = scores_bins.tolist()
    output["occ_hist"] = {}
    output["occ_hist"]["n"] = occ_n.tolist()
    output["occ_hist"]["bins"] = occ_bins.tolist()
    output["logp_hist"] = {}
    output["logp_hist"]["n"] = logp_n.tolist()
    output["logp_hist"]["bins"] = logp_bins.tolist()
    output["sigma_hist"] = {}
    output["sigma_hist"]["n"] = sigma_n.tolist()
    output["sigma_hist"]["bins"] = sigma_bins.tolist()
    if occupancies < 100:
        spacing = math.ceil(input.shape[0] / 512.0)
        output["occ_plot"] = np.sort(input[::spacing, 11])[::-1].tolist()

    # save dict to pikle for parallel run
    output_file = output_name + "_temp.pkl"
    meta_file = output_name + "_meta_temp.pkl"
    with open(output_file, 'wb') as f1:
        pickle.dump(output, f1)
    with open(meta_file, 'wb') as f2:
        pickle.dump(metadata, f2)
    # return output, metadata


def generate_plots_relion(parfile, angles=25, defocuses=25):
    """Deprecated from pyp_shape_pr_values.py: not in use."""
    # convert star file to .par file
    frealign_par_file = parfile.replace(".star", ".par")

    # create new star file with rotation parameters

    parameters = project_params.load_pyp_parameters("..")
    pixel_size = (
        float(parameters["scope_pixel"])
        * float(parameters["data_bin"])
        * float(parameters["extract_bin"])
    )

    # read data star file
    selected = np.array([line.split() for line in open(parfile) if ".mrcs" in line])

    # figure out RELION column numbers from star file header
    rlnClassNumber = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnClassNumber" in line
    ][0] - 1
    rlnVoltage = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnVoltage" in line
    ][0] - 1
    rlnDefocusU = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnDefocusU" in line
    ][0] - 1
    rlnDefocusV = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnDefocusV" in line
    ][0] - 1
    rlnDefocusAngle = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnDefocusAngle" in line
    ][0] - 1
    rlnAngleRot = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnAngleRot" in line
    ][0] - 1
    rlnAngleTilt = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnAngleTilt" in line
    ][0] - 1
    rlnAnglePsi = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnAnglePsi" in line
    ][0] - 1
    rlnOriginX = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnOriginX" in line
    ][0] - 1
    rlnOriginY = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnOriginY" in line
    ][0] - 1
    rlnGroupNumber = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnGroupNumber" in line
    ][0] - 1
    rlnImageName = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnImageName" in line
    ][0] - 1
    rlnMaxValueProbDistribution = [
        int(line.split()[1].replace("#", ""))
        for line in open(parfile)
        if "_rlnMaxValueProbDistribution" in line
    ][0] - 1

    # produce frealign parameter file
    f = open(frealign_par_file, "w")
    f.writelines(metaio.frealign_parfile.NEW_PAR_HEADER)

    for i in range(selected.shape[0]):
        f.write(
            metaio.frealign_parfile.NEW_PAR_STRING_TEMPLATE
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
                float(selected[i, rlnMaxValueProbDistribution]),
                0,
            )
        )
        f.write("\n")

    f.close()

    # generate plots
    generate_plots(frealign_par_file, angles, defocuses, True)
    # os.remove( frealign_par_file )


def plot_local_alignment(
    parameters, name, local_particle, clean_shifts, micrograph_drift, ctf_local
):
    """Plot local alignment results from sprswarm."""
    # plot vector field on image
    display_binning = 8
    com = "{0}/bin/mtffilter -lowpass .0075,.0025 {1}.avg {1}_mtf.mrc".format(
        get_imod_path(), name
    )
    com = "{0}/bin/mtffilter -lowpass .025,.0025 {1}.avg {1}_mtf.mrc".format(
        get_imod_path(), name
    )
    run_shell_command(com)
    com = "{0}/bin/newstack {1}_mtf.mrc {1}_small.mrc -bin {2}".format(
        get_imod_path(), name, display_binning
    )
    run_shell_command(com)
    os.remove(name + "_mtf.mrc")
    micrograph = imageio.mrc.read(name + "_small.mrc")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    from scipy import ndimage

    mmean = micrograph.mean()
    mstd = 2.5 * micrograph.std()
    ax.imshow(micrograph, cmap=cm.Greys_r, vmin=mmean - mstd, vmax=mmean + mstd)
    # white background for figures
    # ax.imshow( micrograph.max() * np.ones( micrograph.shape ), cmap=cm.Greys_r, vmin=mmean-mstd, vmax=mmean+mstd )
    boxx = np.loadtxt("{}.boxx".format(name), ndmin=2)
    box = boxx[
        np.logical_and(boxx[:, 4] == 1, boxx[:, 5] >= int(parameters["extract_cls"]))
    ]
    x = box[:, 0] + box[:, 2] / 2.0
    y = box[:, 1] + box[:, 3] / 2.0
    pointsize = float(parameters["particle_rad"]) / 75.0 * 1000.0 * 8 / display_binning
    # ax.scatter( x/8, y/8, marker="o", s=pointsize, facecolors='none', edgecolors='lime', linewidths=2 )

    # plot local defocus variations
    if parameters["ctf_use_lcl"]:
        from matplotlib.patches import Ellipse

        defocuses = (ctf_local[:, :2] - ctf_local[:, :2].min()) / (
            ctf_local[:, :2].max() - ctf_local[:, :2].min()
        )
        defocuses = 8 / display_binning * (20 + 20 * defocuses)
        # for figures
        # defocuses = 8 / display_binning * ( 40 + 40 * defocuses )

        ells = [
            Ellipse(
                xy=(x[i] / display_binning, y[i] / display_binning),
                width=defocuses[i, 0],
                height=defocuses[i, 1],
                angle=ctf_local[i, 2],
            )
            for i in local_particle
        ]
        cc_colors = ctf_local[:, 3]
        cc_colors = (cc_colors - cc_colors.min()) / (cc_colors.max() - cc_colors.min())
        count = 0
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            # e.set_alpha(.75)
            e.set_linewidth(3)
            # e.set_edgecolor('lime')
            e.set_edgecolor("white")
            e.set_facecolor("none")
            count += 1
    else:
        ax.scatter(
            x / display_binning,
            y / display_binning,
            marker="o",
            s=3 * pointsize,
            facecolors="none",
            edgecolors="white",
            linewidths=4,
            alpha=1,
        )

    # ax.scatter( x/8, y/8, marker="o", s=pointsize, alpha=0.5, c=colors, edgecolors='lime', linewidths=2, cmap=cm.coolwarm )
    velocity_scale = 25.0 / np.linalg.norm(clean_shifts[:, :, -2:], axis=2).mean()

    for particle in range(box.shape[0]):

        if True:
            # movement direction
            velocity = velocity_scale * (
                clean_shifts[particle, -1, -2:] - clean_shifts[particle, 0, -2:]
            )
            local = clean_shifts[particle, :, -2:] + micrograph_drift[:, -2:]
            local -= local[0, :]
            local *= 50.0 / np.linalg.norm(local).mean()
            c = np.linspace(0, local.shape[0] - 1, local.shape[0])
            ax.scatter(
                local[:, 0] + x[particle] / display_binning,
                -local[:, -1] + y[particle] / display_binning,
                c=c,
                cmap=cm.autumn,
                linewidths=0,
                s=25,
                alpha=1,
            )
            ax.annotate(
                particle,
                (x[particle] / display_binning, y[particle] / display_binning),
                xytext=(
                    x[particle] / display_binning - 30 * 8 / display_binning,
                    y[particle] / display_binning + 30 * 8 / display_binning,
                ),
                bbox=dict(boxstyle="round,pad=0.1", fc="yellow", alpha=0.5),
                fontsize=10,
            )

        else:
            # for paper figures
            velocity = velocity_scale * (
                clean_shifts[particle, -1, -2:] - clean_shifts[particle, 0, -2:]
            )
            # ax.arrow( x[particle]/8, y[particle]/8, velocity[0], velocity[-1], head_width=10, head_length=10, fc='lime', ec='lime')
            local = clean_shifts[particle, :, -2:] + micrograph_drift[:, -2:]
            local -= local[0, :]
            local *= 300.0 / np.linalg.norm(local).mean()
            c = np.linspace(0, local.shape[0] - 1, local.shape[0])
            # reversed x-dimension for figures
            ax.scatter(
                -local[:, 0] + x[particle] / display_binning,
                local[:, -1] + y[particle] / display_binning,
                c=c,
                cmap=cm.autumn,
                linewidths=0,
                s=25,
                alpha=1,
            )

    # ax.set_aspect('equal')
    # plt.axis('equal')
    # ax.set_xlim([0,micrograph.shape[0]+50])
    # ax.set_ylim([0,micrograph.shape[1]+10])
    a = plt.gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    ax.set_axis_off()
    plt.axis("off")
    # extent = ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    # plt.title( 'Drift scale = %.2f' % ( 10.0 / velocity_scale ) )
    plt.savefig("{0}_field.pdf".format(name), bbox_inches="tight", pad_inches=0)
    plt.close()


def tomo_slicer_gif(tomogram, output, flipyz=True, averagezslices=8, verbose=False):
    """tomo_slicer_gif - Generate a GIF animation from a tomogram navigating through its z slices 

    Parameters
    ----------
    tomogram : str
        Filename of tomogram, which is typically the one generated by PYP 
    output : str
        Filename of output GIF
    flipyz : bool, optional
        Whether or not Y & Z axis of the tomogram need to be flipped, by default True
    dimensions : int, optional
        Dimensions of output GIF, by default 256
    """
    
    extension = Path(tomogram).suffix

    tomogram_avg = tomogram.replace(extension,".avg")
    if averagezslices > 1:
        command = f"{get_imod_path()}/bin/binvol {tomogram} {tomogram_avg}"
        if flipyz:
            command += f" -ybinning {averagezslices} -xbinning 1 -zbinning 1"
        else:
            command += f" -zbinning {averagezslices} -ybinning 1 -xbinning 1"
        
        run_shell_command(command, verbose=verbose)
    else:
        os.symlink( tomogram, tomogram_avg)

    # flip yz - convert tomogram from 512 x 256 x 512 to 512 x 512 x 256
    tomogram_flip = tomogram.replace(extension, "_flip.rec")
    if flipyz:
        command = "{0}/bin/clip flipyz {1} {2}".format(
            get_imod_path(), tomogram_avg, tomogram_flip
        )
        run_shell_command(command, verbose=verbose)
    else:
        os.symlink( tomogram_avg, tomogram_flip )
 
    # get the mean and std from original tomogram 
    volume = imageio.mrc.read(tomogram_flip)
    dimensions = volume.shape # array shape z, y, x
    num_z_slices = dimensions[0]
    mean, std = volume.mean(), volume.std()
    min_cutoff, max_cutoff = mean - 2 * std, mean + 2 * std

    
    # generate pngs for the middle slices
    starting_slice, ending_slice = 0, num_z_slices - 1
    output_pattern = tomogram_flip.replace(".rec", "")
    check_env()
    command = "{0}/bin/mrc2tif -z {1},{2} -p -S {3},{4} {5} {6}".format(
        get_imod_path(),
        starting_slice,
        ending_slice,
        min_cutoff,
        max_cutoff,
        tomogram_flip,
        output_pattern,
    )
    run_shell_command(command, verbose=verbose)

    # sorting the pngs and create a loop by appending a reverse list
    pngList = [
        png
        for png in os.listdir(".")
        if png.startswith(Path(output_pattern).name) and png.endswith(".png")
    ]
    # pngList.sort(key=lambda x: int(x.replace(".png", "").split(".")[-1]))
    pngList.sort()
    # loop_pngList = pngList + pngList[-2:0:-1]

    # generate a GIF using these pngs
    square_size = int(math.ceil(math.sqrt(len(pngList))))
    if max(dimensions[1:]) * square_size > 16383:
        rec_output = output.replace(".webp",".png")
    else:
        rec_output = output
    command = "/usr/bin/montage -resize {0}x{1} -geometry +0+0 -tile {2}x {3} {4}".format(
        dimensions[2], dimensions[1], square_size, " ".join(pngList), rec_output
    )
    run_shell_command(command, verbose=verbose)

    # clean up pngs and flipped tomogram
    [os.remove(png) for png in pngList]
    os.remove(tomogram_avg)
    os.remove(tomogram_flip)

    # generate central slice
    rec = imageio.mrc.read(tomogram)
    # x, z, y shape of rec in array shape is y,z,x
    rec_x = rec.shape[2]
    rec_z = rec.shape[1]
    avg_start = rec_z // 2 - 5
    avg_end = rec_z // 2 + 5
    
    imageio.writepng(np.average(rec[:, avg_start:avg_end, :], 1), "image.png")
    # commands.getstatusoutput('{0}/convert image.png -contrast-stretch 1%x98% image.png'.format( os.environ['IMAGICDIR'] ) )
    contrast_stretch("image.png")

    # compose quad-plot
    command = "{0}/convert image.png {1}".format(
        os.environ["IMAGICDIR"], output.replace("_rec.webp",".webp")
    )
    run_shell_command(command, verbose=verbose)

    # generate side projections
    x_middle = rec_x // 2
    
    imageio.writepng(np.average(rec[:x_middle, :, :], 0), "side1.png")
    imageio.writepng(np.average(rec[x_middle + 1:, :, :], 0), "side2.png")
    contrast_stretch("side1.png")
    contrast_stretch("side2.png")
    run_shell_command(
        "%s/montage side1.png side2.png -mode concatenate -tile 1x %s"
        % ( os.environ["IMAGICDIR"], output.replace("_rec.webp","_sides.webp") ),
        verbose=verbose,
    )
    # clean up side pngs
    pngList = [
        png
        for png in os.listdir(".")
        if png.startswith("side") and png.endswith(".png")
    ]
    [os.remove(png) for png in pngList]


def tomo_montage(input, output, dimensions=384, verbose=False):
    """tomo_slicer_gif - Generate a montage from an mrc file

    Parameters
    ----------
    tomogram : str
        Filename of tomogram, which is typically the one generated by PYP 
    output : str
        Filename of output GIF
    flipyz : bool, optional
        Whether or not Y & Z axis of the tomogram need to be flipped, by default True
    dimensions : int, optional
        Dimensions of output GIF, by default 256
    """

    # get the mean and std from original tomogram
    volume = imageio.mrc.read(input)
    num_z_slices = volume.shape[0]
    mean, std = volume.mean(), volume.std()
    min_cutoff, max_cutoff = mean - 2 * std, mean + 2 * std

    current_dir = os.getcwd()
    working_dir = input + "_dir"
    os.mkdir( working_dir )
    os.chdir( working_dir )

    # generate pngs
    check_env()
    output_pattern = Path(input).stem
    command = "{0}/bin/mrc2tif -p -S {1},{2} {3} {4}".format(
        get_imod_path(),
        min_cutoff,
        max_cutoff,
        os.path.join( current_dir, input ),
        output_pattern,
    )
    run_shell_command(command, verbose=verbose)

    # sorting the pngs and create a loop by appending a reverse list
    pngList = [
        png
        for png in os.listdir(".")
        if png.startswith(output_pattern) and png.endswith(".png")
    ]
    # pngList.sort(key=lambda x: int(x.replace(".png", "").split(".")[-1]))
    pngList.sort()

    # generate a GIF using these pngs
    tiles = int(math.ceil(math.sqrt(len(pngList))))
    command = "/usr/bin/montage -resize {0}x{0} -geometry +0+0 -tile {1}x {2} {3}".format(
        dimensions, tiles, " ".join(pngList), os.path.join( current_dir, output )
    )
    run_shell_command(command, verbose=verbose)
 
    # clean up pngs and flipped tomogram
    [os.remove(png) for png in pngList]
    os.chdir( current_dir )
    shutil.rmtree( working_dir )

def plot_tomo_ctf(name,verbose=False):
    """Plot CTF function called from tomo_swarm."""
    # Slice of binned reconstruction for quad plot
    if os.path.isfile("ctffind3.png"):

        rec = imageio.mrc.read("{0}.rec".format(name))
        rec_z = rec.shape[1]
        avg_start = rec_z // 2 - 5
        avg_end = rec_z // 2 + 5
        
        imageio.writepng(np.average(rec[:, avg_start:avg_end, :], 1), "image.png")
        # commands.getstatusoutput('{0}/convert image.png -contrast-stretch 1%x98% image.png'.format( os.environ['IMAGICDIR'] ) )
        contrast_stretch("image.png")

        # compose quad-plot
        img2webp("image.png",f"{name}.webp")

def plot_spr_ctf(name, verbose=False):
    # resize image to have 512 pixels in height
    command = "{0}/convert {1}.jpg -flip -resize x512 +append image.png".format(
        os.environ["IMAGICDIR"], name
    )
    run_shell_command(command, verbose=verbose)
    # add frame trajectory to power spectrum image
    if os.path.exists(name + ".xf"):
        plot_trajectory(name)
        command = "{0}/composite -geometry +256+0 {1}_xf.png ctffind3.png ctffind3.png".format(
            os.environ["IMAGICDIR"], name,
        )
        run_shell_command(command, verbose=verbose)
    # add correlation plot
    if os.path.exists("%s.ccc" % name):
        plot_ccc(name)
        command = "{0}/composite -geometry +256+256 {1}_ccc.png ctffind3.png ctffind3.png".format(
            os.environ["IMAGICDIR"], name,
        )
        run_shell_command(command, verbose=verbose)
    # create montage
    command = "{0}/montage image.png {1}_CTFprof.png ctffind3.png -geometry x512 -geometry +0+0 {1}_view.webp".format(
        os.environ["IMAGICDIR"], name
    )
    run_shell_command(command, verbose=verbose)


def plot_trajectories(
    name_png,
    ali_path,
    box,
    clean_regularized_shifts,
    parameters,
    is_tomo=False,
    rotation=False,
    savefig=False,
):
    
    name = ali_path.strip(".mrc")
    scratch_name = os.path.basename(name)

    
    assert(os.path.exists(f"{name}.avg")), f"Cannot find micrograph {name}.avg"

    # only draw 0 degree tilt 
    if "tomo" in parameters["data_mode"].lower():
        try:
            tilts = np.loadtxt(f"{name}.tlt")
        except:
            metadata = pyp_metadata.LocalMetadata(f"{name}.pkl").data
            tilts = metadata["tlt"].to_numpy()

        zero_tilt_index = np.argmin(abs(tilts))

        com = "{0}/bin/newstack -secs {1} {2}.avg {2}.avg~ && mv {2}.avg~ {2}.avg".format(
                get_imod_path(), zero_tilt_index, scratch_name
        )
        run_shell_command(com)

    display_binning = 8
    com = "{0}/bin/mtffilter -lowpass .0075,.0025 {1}.avg {1}_mtf.mrc".format(
        get_imod_path(), scratch_name
    )
    com = "{0}/bin/mtffilter -lowpass .025,.0025 {1}.avg {1}_mtf.mrc".format(
        get_imod_path(), scratch_name
    )
    run_shell_command(com)

    com = "{0}/bin/newstack {1}_mtf.mrc {1}_small.mrc -bin {2}".format(
        get_imod_path(), scratch_name, display_binning
    )
    run_shell_command(com)

    os.remove(scratch_name + "_mtf.mrc")
    micrograph = imageio.mrc.read(scratch_name + "_small.mrc")

    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    mmean = micrograph.mean()
    mstd = 2.5 * micrograph.std()
    ax.imshow(micrograph, cmap=cm.Greys_r, vmin=mmean - mstd, vmax=mmean + mstd)

    x = box[:, 0]
    y = box[:, 1]

    if rotation:
        clean_shifts = clean_regularized_shifts[:, :, 1:3]
    else:
        clean_shifts = clean_regularized_shifts[:, :, -2:]

    local_trajectories = clean_shifts[:, :, -2:]
    scale = get_scale_for_trajectory(local_trajectories, box[:, :2])

    for i, particle in enumerate(range(box.shape[0])):

        local = local_trajectories[particle, :, :]

        local *= scale if np.linalg.norm(local).mean() > 0.0 else 0.0
        c = np.linspace(0, local.shape[0] - 1, local.shape[0])

        # plot the dots for the trajectory
        ax.scatter(
            (local[:, 0] + x[particle]) / display_binning,
            (local[:, -1] + y[particle]) / display_binning,
            c=c,
            cmap=cm.autumn,
            linewidths=0,
            s=0.18*scale,
            alpha=1,
        )
        # plot the line
        ax.plot(
            (local[:, 0] + x[particle]) / display_binning,
            (local[:, -1] + y[particle]) / display_binning,
            c="g",
            linewidth=0.04*scale,
        )

    a = plt.gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    ax.set_axis_off()
    plt.axis("off")

    if savefig:
        plt.savefig("{0}.pdf".format(name_png), bbox_inches="tight", pad_inches=0)
    plt.close()

    # workaround ImageMagic's error caused by security policy used to convert PDF files
    run_shell_command(f"gs -dSAFER -r500 -sDEVICE=pngalpha -o {name_png}.png {name_png}.pdf")
    run_shell_command(f"convert {name_png}.png {name_png}_local.webp")


def histogram_particle_tomo(scores: list, threshold: float, tiltseries: str, save_path: str):

    # remove -1 (i.e. particles that do not have enough projections, missing in parifle etc.)
    scores = [_ for _ in scores if _ >= 0.0]
    
    if len(scores) > 0:
        good_scores = [_ for _ in scores if _ >= 0.0 and _ >= threshold]
        bad_scores = [_ for _ in scores if _ >= 0.0 and _ < threshold]

        bins = 80
        interval = (max(scores) - min(scores)) / bins 
        good_bins = int((max(good_scores) - min(good_scores)) / interval) if len(good_scores) > 0 else 0
        bad_bins = int((max(bad_scores) - min(bad_scores)) / interval) if len(bad_scores) > 0 else 0

        fig, axs =  plt.subplots(1, 1, tight_layout=True)
        axs.set_xlim([-0.5, max(max(scores)+1, threshold)])
        if len(good_scores) > 0 and good_bins > 0:
            axs.hist(good_scores, bins=good_bins, color="royalblue", alpha=1.0)
        if len(bad_scores) > 0 and bad_bins > 0:
            axs.hist(bad_scores, bins=bad_bins, color="royalblue", alpha=0.5)

        axs.set_xlabel('Mean score', fontsize=12, labelpad=10)
        axs.set_ylabel('Population', fontsize=12, labelpad=10)
        axs.spines.right.set_visible(False)
        axs.spines.top.set_visible(False)
        axs.spines["bottom"].set_linewidth(1)
        axs.spines["left"].set_linewidth(1)
        axs.tick_params(axis='both', which='major', labelsize=10)

        plt.axvline(x=threshold, linestyle="dashed", linewidth=2.0, color='black')
        plt.savefig(f"{save_path}/{tiltseries}_scores.svgz")
        plt.close()
    else:
        logger.warning("No valid projections for tilt-series %s are left" % tiltseries)



def get_scale_for_trajectory(local_trajectories, coordinates) -> float:

    # get the mean length for each trajectory
    max_lengths = [np.max(distance.cdist(dots, dots, 'euclidean')) for dots in local_trajectories]
    mean_length = np.mean(max_lengths)

    # get the mean shortest distance for each particle
    dists_particles = distance.cdist(coordinates, coordinates, 'euclidean')

    min_dists = [min([dist for dist in particle if dist > 0.0]) for particle in dists_particles]
    dist = np.percentile(min_dists, 75)

    # 0.6 is meant to leave some space, so it won't look too crowded
    if mean_length > 0:
        return dist / mean_length * 0.6
    else:
        return 1


def par2bild(parfile, output, parameters):
    # Read angles parameters from parfile and convert to .bild file to view in ChimeraX


    if "tomo" in parameters["data_mode"]:
        tilt_max = parameters["reconstruct_maxtilt"]
        is_tomo = f"--tomo --tilt_max {tilt_max}"
    else:
        is_tomo = ""

    comm= os.environ["PYP_DIR"] + f"/external/postprocessing/par_to_bild.py --input '{parfile}' --output '{output}' {is_tomo} --apix {parameters['scope_pixel']*parameters['data_bin']*parameters['extract_bin']} --healpix_order 4 --boxsize {parameters['extract_box']} --height_scale 0.3 --width_scale 0.5 --occ_cutoff {parameters['reconstruct_cutoff']} --sym {parameters['particle_sym']} "

    run_shell_command(comm, verbose=False)
    
    if not os.path.exists(output):
        logger.error(f"Failed to produce {output}")
