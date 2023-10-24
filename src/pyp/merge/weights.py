import numpy as np


def fft2D_to_rfft2(w):
    sy = w.shape[0]
    sx = w.shape[1]
    faux = np.zeros([sy, sx // 2 + 1])
    faux[:, 0 : sx // 2] = w[:, sx // 2 : sx]
    faux[:, sx // 2] = 1.0
    # ATENTION: si uso esta function para ir del fft2D al rfft2
    # descomentar esta linea y comentar la anterior. Lo de fijar
    # esa columan en 1.0 es para imponer que los weights sean
    # simetricos y no introduzcan artefactos en la fft.
    # faux[:,sx/2] = w[:,0]
    fout = np.zeros([sy, sx // 2 + 1])
    fout[0 : sy // 2, :] = faux[sy // 2 : sy, :]
    fout[sy // 2 : sy, :] = faux[0 : sy // 2, :]
    return fout


def motion_weights_aux(sx, sy):
    fx = np.power(np.linspace(-0.5, 0.5, sx + 1), 2)
    fy = np.power(np.linspace(-0.5, 0.5, sy + 1), 2)
    sfx = len(fx)
    sfy = len(fy)
    expX = np.zeros([1, sfx])
    expY = np.zeros([sfy, 1])
    for i in range(sfx):
        expX[0, i] = np.exp(-fx[i])
    for i in range(sfy):
        expY[i, 0] = np.exp(-fy[i])
    expX = np.tile(expX, [sy + 1, 1])
    expY = np.tile(expY, [1, sx + 1])
    return expX, expY


def motion_weights(radiusX, radiusY, sx, sy):
    invSigmaX2 = 1 / radiusX
    invSigmaY2 = 1 / radiusY

    expX, expY = motion_weights_aux(sx, sy)

    expX = np.power(expX, invSigmaX2)
    expY = np.power(expY, invSigmaY2)
    w = expX * expY
    return w


def radDamage_weights_aux(sx, sy):
    bins = np.linspace(0, 0.5 * np.sqrt(2), 4 * np.max([sx, sy]))
    Ne = np.exp(-bins)

    faux = np.zeros([sy // 2 + 1, sx // 2 + 1])
    fx = np.power(np.linspace(0, 0.5, sx // 2 + 1), 2)
    fy = np.power(np.linspace(0, -0.5, sy // 2 + 1), 2)
    sfx = len(fx)
    sfy = len(fy)
    for i in range(sfy):
        for j in range(sfx):
            r = np.sqrt(fy[i] + fx[j])
            faux[i, j] = np.interp(r, bins, Ne)

    fout = np.zeros([sy + 1, sx + 1])
    fout[sy // 2 : sy + 1, sx // 2 : sx + 1] = faux
    fout[sy // 2 : sy + 1, 0 : sx // 2 + 1] = np.fliplr(faux)
    fout[0 : sy // 2 + 1, :] = np.flipud(fout[sy // 2 : sy + 1, :])

    # remove NaNs
    fout[0, 0] = fout[0, 1]
    fout[sy, 0] = fout[sy, 1]
    fout[0, sx] = fout[0, sx - 1]
    fout[sy, sx] = fout[sy - 1, sx - 1]
    return fout


def radDamage_weights(delta, deltaF, Nframe, sx, sy):

    values = radDamage_weights_aux(sx, sy)

    # print 'Values =', values.min(), values.max()
    Ne = np.power(values, delta)

    switch_value = 0.0025
    fraction = delta
    delta = 37
    switch_value = np.power(np.exp(-0.5 * fraction), delta)

    # hard transition
    if False:
        Ne = np.maximum(Ne, np.ones(Ne.shape) * switch_value)
    # soft transition
    else:
        switch = np.power(switch_value, 1.0 / delta)
        # print switch, np.power( switch, delta )
        slope = 0.05  # ( smaller = steeper )
        Sx = 0.5 * (1 + np.tanh((values - switch) / slope))
        Ne = Sx * Ne + (1 - Sx) * switch_value

    w = np.exp(-deltaF * np.power(Nframe, 4) / Ne)
    # print("In radDamage_weights, output shape: ", w.shape)
    return w


def set_motion_parameters(xf_file, factor, radiK):
    px, py = compute_speed(xf_file)

    k1 = 0.3
    radiusX = -np.power((factor * radiK) / (np.abs(px) + 1), 2) / np.log(k1)
    radiusY = -np.power((factor * radiK) / (np.abs(py) + 1), 2) / np.log(k1)
    return radiusX, radiusY


def compute_speed(xf_file):
    trans = np.loadtxt(xf_file)
    tx = trans[:, 4]
    ty = trans[:, 5]
    N = len(tx)
    # use centered differences
    speedX = ((tx[1 : N - 1] - tx[0 : N - 2]) + (tx[2:N] - tx[1 : N - 1])) / 2
    speedY = ((ty[1 : N - 1] - ty[0 : N - 2]) + (ty[2:N] - ty[1 : N - 1])) / 2

    # use simple backward differences
    # speedX=tx[2:N]-tx[1:N-1]
    # speedY=ty[2:N]-ty[1:N-1]

    speedXout = np.zeros(N)
    speedYout = np.zeros(N)

    speedXout[1 : N - 1] = speedX
    speedXout[0] = speedX[0]
    speedXout[N - 1] = speedX[N - 3]

    speedYout[1 : N - 1] = speedY
    speedYout[0] = speedY[0]
    speedYout[N - 1] = speedY[N - 3]
    return speedXout, speedYout


def combined_weights(
    radiusX, radiusY, delta, deltaF, Nframe, sx, sy, radiation_only=False
):

    wr = radDamage_weights(delta, deltaF, Nframe, sx, sy)

    if radiation_only:
        w = wr
    else:
        wm = motion_weights(radiusX, radiusY, sx, sy)
        w = wm * wr

    # back to original size
    w = w[0:sy, 0:sx]
    # conver to rfft2 format
    wout = fft2D_to_rfft2(w)

    # print("In combined_weights, output shape", wout.shape)
    return wout


def combined_weights_movie(
    xf_file, radiK, delta, deltaF, sx, sy, binFactor, scores=np.empty([0])
):

    if len(scores) > 0 and radiK == 0:
        N = len(scores)
        radiusX = radiusY = np.empty([N])
    else:
        radiusX, radiusY = set_motion_parameters(xf_file, binFactor, radiK)
    if len(scores) > 0:
        N = len(scores)
    else:
        N = len(radiusX)
    w = np.zeros([N, sy, sx // 2 + 1])
    w_acum = np.zeros([sy, sx // 2 + 1])
    if len(scores) > 0:
        if False:
            # radiation damage plus measured weights
            weights = (
                1.0 * np.arange(N) / N
                + 1
                - (scores - scores.min()) / (scores.max() - scores.min())
            )
        else:
            # measured weights
            weights = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            weights = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        weights = 1.0 * np.arange(N) / N
        # weights = 1.0 * np.arange(N)

    for f in range(N):
        w[f] = combined_weights(
            radiusX[f], radiusY[f], delta, deltaF, weights[f], sx, sy, radiK == 0
        )
        w_acum += w[f]

    # Normalize the weights
    for f in range(N):
        w[f] = w[f] / w_acum

    # print("In main combined_weights_movie fn, output shape: ", w.shape)
    return w
