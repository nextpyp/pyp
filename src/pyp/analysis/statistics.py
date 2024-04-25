import numpy as np
import pandas as pd

from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


def gauss_function(x, amp, x0, sigma):
    return amp * np.exp(-((x - x0) ** 2.0) / (2.0 * sigma ** 2.0))


def optimal_threshold(samples, criteria="optimal", plot_results=""):
    """Find optimal threshold using Gaussian Mixture Model.
    
    Posible criteria are:
          'mean'   :(of means)
          'min'    :(of sum of Gaussians),
          'opt'    :(intersection of both Gaussians)
    """

    # deal with case of empty samples
    if np.var(samples) == 0:
        return 1

    # Fit GMM
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=2, covariance_type="full", tol=1e-6, reg_covar=1e-6
    )
    gmm = gmm.fit(X=np.expand_dims(samples, 1))

    # Evaluate GMM
    gmm_x = np.linspace(samples.min(), samples.max(), 5000)
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

    # Construct function manually as sum of gaussians
    gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
    for m, c, w in zip(
        gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()
    ):
        gauss = gauss_function(x=gmm_x, amp=1, x0=m, sigma=np.sqrt(c))
        gmm_y_sum += gauss / np.trapz(gauss, gmm_x) * w

    mean = np.mean(gmm.means_)

    m1 = gmm.means_[0]
    m2 = gmm.means_[1]
    s1 = gmm.covariances_[0]
    s2 = gmm.covariances_[1]
    p1 = gmm.weights_[0]
    p2 = gmm.weights_[1]

    m1x = np.argmin(np.fabs(gmm_x - m2))
    m2x = np.argmin(np.fabs(gmm_x - m1))

    if m1x > m2x:
        m = m2x
        m2x = m1x
        m1x = m

    minimum = m1x + np.argmin(gmm_y_sum[m1x:m2x])

    gauss = gauss_function(x=gmm_x, amp=1, x0=m1, sigma=np.sqrt(s1))
    gmm_1 = (gauss / np.trapz(gauss, gmm_x) * p1).squeeze()

    gauss = gauss_function(x=gmm_x, amp=1, x0=m2, sigma=np.sqrt(s2))
    gmm_2 = (gauss / np.trapz(gauss, gmm_x) * p2).squeeze()

    optimal = np.argmin(np.fabs(gmm_1[m1x:m2x] - gmm_2[m1x:m2x]))
    optimalx = gmm_x[m1x + optimal]

    if (
        not "_scores.png" in plot_results
        and (
            (gmm_1[m1x + optimal - 1] - gmm_2[m1x + optimal - 1])
            * (gmm_1[m1x + optimal + 1] - gmm_2[m1x + optimal + 1])
            > 0
        )
        or (
            gmm_y_sum[m1x + optimal] > gmm_1.max()
            and gmm_y_sum[m1x + optimal] > gmm_2.max()
        )
    ):
        gmm = GaussianMixture(
            n_components=1, covariance_type="full", tol=1e-6, reg_covar=1e-6
        )
        gmm = gmm.fit(X=np.expand_dims(samples, 1))
        threshold = (gmm.means_[0] - 3.0 * np.sqrt(gmm.covariances_[0]))[0][0]
    elif "opt" in criteria:
        threshold = optimalx
    elif "min" in criteria:
        threshold = minimum
    else:
        threshold = mean

    if len(plot_results) > 0:

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="darkgrid")
        # Make regular histogram
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
        ax.hist(samples, bins=50, density=True, alpha=0.5, color="#0070FF")
        # ax.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")
        ax.plot(
            gmm_x, gmm_y_sum, color="black", lw=2, label="Gauss_sum", linestyle="dashed"
        )

        ax.plot(
            gmm_x,
            gmm_1,
            color="red",
            lw=1,
            label="GMM 1 ({0},{1},{2})".format(m1, s1, p1),
        )
        ax.plot(
            gmm_x,
            gmm_2,
            color="blue",
            lw=1,
            label="GMM 2 ({0},{1}),{2}".format(m2, s2, p2),
        )

        # ax.plot( mean * np.ones( gmm_x.shape ), gmm_y, label='mean = %f' % mean )
        # ax.plot( gmm_x[minimum] * np.ones( gmm_x.shape ), gmm_y, label='minimum = %f' % gmm_x[minimum] )

        # ax.plot( optimalx * np.ones( gmm_x.shape ), gmm_y, label='optimal = %f' % optimalx )

        ax.plot(
            (threshold * np.ones(gmm_x.shape)).squeeze(),
            gmm_y,
            color="green",
            label="Threshold = %f" % optimalx,
        )

        # Annotate diagram
        ax.set_ylabel("Probability density")
        ax.set_xlabel("Arbitrary units")
        # ax.set_xlabel("Particle Score (against 3D reference)")
        # ax.set_xlabel("CTFFIND4 score")
        # ax.set_xlim([10,35])

        # Make legend
        plt.legend()
        plt.savefig(plot_results, bbox_inches="tight")

    # check if there IS intersection between the means
    return threshold


def optimal_threshold_bayes(samples, criteria="optimal", plot_results=""):

    # Fit GMM
    from sklearn.mixture import GaussianMixture

    components = 3

    # for component in range(1,components):
    if True:
        gmm = GaussianMixture(
            n_components=components, covariance_type="full", tol=1e-6, reg_covar=1e-6
        )
        X = np.expand_dims(samples, 1)
        gmm = gmm.fit(X)
        # print components, gmm.bic(X)
    #    break

    # print gmm.means_, gmm.covariances_

    # Evaluate GMM
    gmm_x = np.linspace(samples.min(), samples.max(), 50)
    gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

    # Construct function manually as sum of gaussians
    gmm_y_sum = np.full_like(gmm_x, fill_value=0, dtype=np.float32)
    for m, c, w in zip(
        gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()
    ):
        gauss = gauss_function(x=gmm_x, amp=1, x0=m, sigma=np.sqrt(c))
        gmm_y_sum += gauss / np.trapz(gauss, gmm_x) * w

    if len(plot_results) > 0:

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(style="darkgrid")
        # Make regular histogram
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
        ax.hist(samples, bins=50, density=True, alpha=0.5, color="#0070FF")
        # ax.plot(gmm_x, gmm_y, color="crimson", lw=4, label="GMM")

        ax.plot(
            gmm_x, gmm_y_sum, color="black", lw=2, label="Gauss_sum", linestyle="dashed"
        )

        h, edges = np.histogram(samples, 50)

        logger.info("difference = %f", np.abs(h - gmm_y_sum).mean())

        for i in range(components):

            gauss = gauss_function(
                x=gmm_x, amp=1, x0=gmm.means_[i], sigma=np.sqrt(gmm.covariances_[i])
            )
            gmmm = (gauss / np.trapz(gauss, gmm_x) * gmm.weights_[i]).squeeze()
            ax.plot(gmm_x, gmmm, lw=2)

        # Annotate diagram
        ax.set_ylabel("Probability density")
        ax.set_xlabel("Arbitrary units")

        # Make legend
        plt.legend()
        plt.savefig(plot_results, bbox_inches="tight")

    return 0


def weighted_by_tilt_angle(ptl_data, tltang_dict):
    """Calculate weighted frame occ and logp according to tilt angles"""
    # occ = ptl_occ_tltang[:, 0].ravel()
    logp = ptl_data[:, 2].ravel()
    # sigma = ptl_occ_tltang[:, 2].ravel()
    tltind = ptl_data[:, -1]
    # tiltangle = []
    tind_angle_dict = {i: tltang_dict[i][0].angle for i in tltang_dict.keys()}
    df_tind = pd.DataFrame(tltind, columns=["Tind"])
    # mapped_angles = np.vectorize(tind_angle_dict.get)(tind_in_film)
    df_tind["Angle"] = df_tind["Tind"].map(tind_angle_dict)
    
    #for tlt_ind in tltind:
    #    proj_tlt_angle  =  tltang_dict[tlt_ind][0].angle
    #    tiltangle.append(proj_tlt_angle)

    tltang = df_tind["Angle"].to_numpy()
    if np.count_nonzero(tltang) > 1:
        max_angle = np.amax(np.abs(tltang))
        gsigma = max_angle / 6
        gauss_weight = []
        for e in tltang:
            gauss_weight.append(gauss_function(e, amp=1, x0=0, sigma=gsigma))
        # occ = np.sum(occ * np.array(gauss_weight)) / np.sum(np.array(gauss_weight))
        logp = np.sum(logp * np.array(gauss_weight)) / np.sum(np.array(gauss_weight))
        # sigma = np.sum(sigma * np.array(gauss_weight)) / np.sum(np.array(gauss_weight))
    else:
        logp = ptl_data[:30, 1].ravel()
        logp = np.mean(logp)

    return logp

def get_class_score_weight(parx3d, score_col, scanord_col):
    """score averages across different tilt range"""
    # get the max score fromm all classes
    
    score_max = np.amax(parx3d[:,:,score_col], axis=0 )
 
    tilt_all = parx3d[0, :, scanord_col]
    tilt_range = np.unique(tilt_all)

    scoreavg_tilt = {}

    for tilt in tilt_range:
        index = np.argwhere(tilt_all == tilt)
       
        score_data = np.sum(score_max[index]) / index.shape[0]

        scoreavg_tilt.update({tilt:score_data})

    return scoreavg_tilt

def weighted_by_scoreavgs(ptl_logp_scanord, scoreavg_tilt):

    """For overlapping situations, score averages may provide more robust validation of projection alignment."""

    scanord = ptl_logp_scanord[:, -1].ravel()
    logp = ptl_logp_scanord[:, 0].ravel()
    weights = []
    for order in scanord:
        score = scoreavg_tilt[order]
        weights.append(score)

    logp = np.sum(logp * np.array(weights)) / np.sum(np.array(weights))
    return logp