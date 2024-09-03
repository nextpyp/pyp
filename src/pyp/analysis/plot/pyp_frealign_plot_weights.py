#!/usr/bin/env python

import numpy
import os
import seaborn as sns

from pyp.analysis import plot
import matplotlib.pyplot as plt
from pyp.system.local_run import run_shell_command

# if not "PYP_DIR" in os.environ:
#     os.environ["PYP_DIR"] = os.environ["HOME"] + "/PYP"
# sys.path.append(os.environ["PYP_DIR"] + "/python")

def plot_weights(name, input, frames, frames_per_tilt, boxsize, pixel):

    A = numpy.loadtxt(input)

    frames = int(A.shape[0] / ((boxsize + 1) * (boxsize / 2)))
    W = numpy.zeros([frames, int(boxsize / 2 + 1), boxsize])

    count = 0
    frame = 0 
    # for frame in range(frames):
    while count < A.shape[0]:
        for j in range(boxsize):
            for i in range(1, int(boxsize / 2 + 1)):
                W[frame, i, j] = A[count]
                count += 1
        for j in range(int(boxsize / 2)):
            i = 0
            W[frame, i, j] = A[count]
            count += 1
        frame += 1

    weights = numpy.swapaxes(W, 2, 1) / W.sum(axis=0).mean()

    # weights = weights[:,:,::-1]

    plot.guinier_plot(
        weights, f"{name}_weights.svgz", pixel
    )

    if os.path.exists("scores.txt"):
        scores = numpy.loadtxt("scores.txt", ndmin=2)
        plot_mean_score_tomo(f'{name}_scores.svgz', scores, num_frames=frames_per_tilt)

        os.rename("scores.txt", f"{name}_scores.txt")


def plot_mean_score_tomo(filename, scores, num_frames=1):

    sns.set(style='darkgrid')

    fig, ax = plt.subplots( 1, 1, figsize=(10,5), dpi=200)
    plt.grid(axis="x", linestyle = '--', linewidth = 0.5)

    x = [_ for _ in range(scores.shape[0])]
    y = [s for s in scores]
    xticks = [_ for _ in range(0, scores.shape[0], num_frames)]

    tilt_x = [x[_] for _ in range(0, len(x), num_frames)]
    tilt_y = [y[_] for _ in range(0, len(x), num_frames)]

    # remove the tilts that do not have postive scores (do not have any projections)
    new_x, new_y, new_xticks, new_tilt_x, new_tilt_y = [], [], [], [], []

    for ind in range(0, scores.shape[0], num_frames):
        score = y[ind]
        if score >= 0.0:
            new_x += x[ind:int(ind+num_frames)]
            new_y += y[ind:int(ind+num_frames)]
            new_xticks.append(xticks[int(ind/num_frames)])
            new_tilt_x.append(tilt_x[int(ind/num_frames)])
            new_tilt_y.append(tilt_y[int(ind/num_frames)])
    
    x, y, xticks, tilt_x, tilt_y  = new_x, new_y, new_xticks, new_tilt_x, new_tilt_y

    ax.scatter(x, y, s=10)
    ax.scatter(tilt_x, tilt_y, s=50, alpha=0.2, color="green")
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(frame / num_frames) for frame in xticks], fontsize=10, rotation = 45)

    ax.set_xlabel('Order of acquisition', fontsize=15, fontweight='semibold', labelpad=10 )
    ax.set_ylabel('Normalized mean score', fontsize=15, fontweight='semibold', labelpad=10 )

    plt.tight_layout()  
    plt.savefig(filename)


def plot_mean_score_tilt(filename, scores, num_frames=1):
    
    ret = numpy.zeros((1, num_frames))
    
    scores = scores.T
    ctr = 1
    for i in range(0, scores.shape[1], num_frames):
        mean = numpy.mean(scores[0][i: i+num_frames])
        ret += scores[0][i: i+num_frames] - mean
        ctr+=1

    ret -= numpy.mean(ret)

    fig, ax = plt.subplots( 1, 1, figsize=(10,5), dpi=100)
    ax.scatter( [_ for _ in range(ret.shape[1])], [ _ for _ in ret], s=50 )
    plt.tight_layout()
    plt.savefig(filename)
