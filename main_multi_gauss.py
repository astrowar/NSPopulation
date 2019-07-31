


import matplotlib.pyplot as plt

#===============================================================
#ROTINA PARA FAZER o PLOT

def plot_hist_i(uuid_named, MMw, xmm, ymm  ,  loss_value):
    # Annotate diagram
    xmin, xmax = np.min(MMw), np.max(MMw)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=[8, 9])
    ax1.hist(MMw, bins=350, density=True, alpha=0.5, range=(xmin, xmax), color="#707000")
    ax2.hist(MMw, bins=350, density=True, alpha=0.5, range=(xmin, xmax), color="#0070FF")
    ax3.hist(MMw, bins=350, density=True, alpha=0.5, range=(xmin, xmax), color="#0070FF")

    ax1.plot(xmm, ymm, color="crimson", lw=1.2)
    ax2.plot(xmm, ymm, color="crimson", lw=1.2)
    ax3.plot(xmm, ymm, color="crimson", lw=1.2)

    plt.ylim(0.0, 1.5)

    ax1.set_xlim(left=xmin, right=xmax)
    ax2.set_xlim(left=0.12, right=0.2)
    ax3.set_xlim(left=xmin, right=xmax)

    ax1.set_ylabel("Probability density")
    ax3.set_xlabel("Mass")
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    ax1.text(right, top, "log likelihood  " + str(loss_value), horizontalalignment='right', verticalalignment='bottom',
             transform=ax1.transAxes)

    # Draw legend
    plt.legend()
    # plt.show()
    plt.savefig("results/" + str(uuid_named) + '.png')
    plt.close()

#====================================================================================

import os
import configparser
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np

import numpy as np
import sys
import uuid

import random

def compute_graph(session, xDist, xmin, xmax):
    # xxrange = tf.range(xmin,xmax, 0.001)
    # print(xmin,xmax)
    # xxrange = tf.range(xa, xb, 0.03)
    # xxrange = tf.range(0.0, 3.1323094594112066, 0.0001)
    xxrange = tf.range(float(xmin), float(xmax), float(abs(xmax - xmin) * 0.0001))
    print(xDist, xxrange)
    yDist = xDist.prob(xxrange)
    ypts = session.run(yDist)
    return list(session.run(xxrange)), list(ypts)



def plot_hist(uuid_named, MMw, session, xDist, loss_value):
      # Annotate diagram
      xmin, xmax = np.min(MMw), np.max(MMw)
      xmm, ymm = compute_graph(session, xDist, xmin, xmax)
      plot_hist_i(uuid_named, MMw,xmm, ymm,  loss_value)



def span_points(x, s, n=100):
    """ converte 1 ponto de amostragem em N pontos centralizados em x com dispersão s """
    if (n == 1):
        return np.array([x])
    # print(s, np.random.randn(4)/10.0 )
    return x + s * np.random.randn(n)


def get_points(X, S, n=100):
    """ obtem os ponto com incertezas embutidas"""
    sample = []
    nn = len(X)
    for i in range(nn):
        xq = span_points(X[i], S[i], n)
        sample.append(xq[xq >= 0])  # adiocnao pontos positivos apenas
        # sample.append(span_points(Mm[i], sigmaM[i], n))
    return np.concatenate(sample)


def print_model_parameters(session, loss, fileOut=sys.stdout):
    var_23 = [v for v in tf.global_variables() if v.name[0] == "x" and len(v.name) == 5]
    print(" x:", session.run(var_23), file=fileOut)

    svar_23 = [v for v in tf.global_variables() if v.name[0] == "s" and len(v.name) == 5]
    print(" s:", session.run(svar_23), file=fileOut)

    mvar_23 = [v for v in tf.global_variables() if v.name[0] == "m" and len(v.name) == 5]

    if len(mvar_23) > 1:
        # print([v for v in tf.global_variables() if v.name[0] == "m" ])
        mvar_23.pop()
        mvar_23.append(1.0 - np.sum(mvar_23))
        print(" m:", session.run(mvar_23), file=fileOut)
    else:
        print(" m: [1.0]", file=fileOut)

    lss = session.run(loss)
    print(" loss ", lss, file=fileOut)


print(tf.__version__)


def BIC(model, Xpts, num_gaussians, err=1e-12):
    """ modelo de BIC """
    n_parameters = (num_gaussians * 3 - 1)

    npts = -1
    if isinstance(Xpts, (np.ndarray)):
        npts = Xpts.get_shape().as_list()[0]
    else:
        if isinstance(Xpts, (tf.Tensor)):
            npts = Xpts.get_shape().as_list()[0]
        else:
            npts = len(Xpts)

    # npts = Xpts.get_shape().as_list()[0]
    logLikMean = tf.reduce_mean(tf.log(model.prob(Xpts) + err))
    # logLikMean = tf.reduce_mean(model.log_prob(Xpts))
    return - 2 * logLikMean + 2 * n_parameters


def get_mix(m, x, s, j, nn):
    x1 = tf.Variable(x, name="x" + str(j) + str(nn))
    # x1 = tf.constant(x, name="x" + str(j) + str(nn))
    s1 = tf.Variable(s, name="s" + str(j) + str(nn))
    m1 = tf.Variable(m, name="m" + str(j) + str(nn))

    comp_1 = tfd.Normal(loc=x1, scale=s1)
    return m1, comp_1
    # return -0.5 * tf.square((XP - x1) / s1) - tf.log(s * 2.50663)


def rec_sum(mm):
    if len(mm) == 1:    return mm[0]
    return tf.add(mm[0], rec_sum(mm[1:]))


def get_normalized_complement(mm):
    if len(mm) == 0:
        return 1.0
    # return  tf.add( 1.0 , -rec_sum(mm) , name="msum")
    return 1.0 - rec_sum(mm)


def get_mixture(j, xoo, sx_defalut ):
    # mi, ni = [ get_mix(0.33, xoo[i], 0.4, j, i) for i in range(len(xoo))]
    mms = []
    nns = []
    value = 1.0 / (1.0 * len(xoo))
    for m, n in [get_mix(value, xoo[i], sx_defalut, j, i) for i in range(len(xoo))]:
        mms.append(m)
        nns.append(n)
    print(mms[:-1])
    mcomp = get_normalized_complement(mms[:-1])
    print(mcomp)
    mms = mms[:-1] + [mcomp]
    print(mms)
    # m2, n2 = get_mix(0.33, 1.3, 0.4, j, 2)
    # m3, n3 = get_mix(0.33, 1.5, 0.4, j, 3)

    xDist = tfd.Mixture(cat=tfd.Categorical(probs=mms), components=nns)
    return xDist

    # return n1+n2+n3


fdata = tf.float32


def gCapper(session, x):
    if x is not None:
        # y =session.run(x)
        # safe_grad = tf.where(tf.is_nan(x), 0.0, x)
        return x
    return x


def train(session, opt, loss, eta):
    # Compute the gradients for a list of variables.
    grads_and_vars = opt.compute_gradients(loss)

    # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
    # need to the 'gradient' part, for example cap them, etc.

    capped_grads_and_vars = [(gCapper(session, gv[0]), gv[1]) for gv in grads_and_vars]
    # Ask the optimizer to apply the capped gradients.
    opt.apply_gradients(capped_grads_and_vars, 0.01)


def compute_gaussian_fit(xo_, Mm_, sigmaM_, sx_default):
    print("start Compute")
    np.random.seed(723188)

    pts = get_points(Mm_, sigmaM_, 10)
    XP = tf.constant(pts, dtype=fdata)

    xDist_1 = get_mixture(1, xo_, sx_default)
    # xDist_2 = get_mixture(2,[1.1,1.3,1.5])
    # xDist_3 = get_mixture(3,[1.2,1.4,1.6])

    # err = tf.Variable(err_value, name="err", trainable=False )
    err = 1e-30
    eta = 1e-6  # learning rate

    learning_rate = tf.placeholder(tf.float32)

    # loss = tf.reduce_mean([tf.reduce_mean(tf.log(xDist.prob(XP) + err)) for xDist in [xDist_1, xDist_2, xDist_3]])

    loss = -tf.reduce_mean(tf.log(xDist_1.prob(XP) + err))
    opt_a = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_a = opt_a.minimize(loss)

    saver = tf.train.Saver()
    model_path_file = "/tmp/model.ckpt"
    uuid_named = uuid.uuid4()
    i = 1
    j = 1
    k = 5
    x00 = ([v for v in tf.global_variables() if v.name[0] == "x"])[0]

    nsubLopp = 10000
    #nsubLopp = 10
    pertub = 0

    with tf.Session() as session:
        # init = tf.initialize_all_variables()
        session.run(tf.global_variables_initializer())
        # session.run(init)
        save_path = saver.save(session, model_path_file)
        old_lss = None
        while i < 500 * nsubLopp:
            q = session.run(train_a, feed_dict={learning_rate: eta})

            # train(session, opt_a ,  loss , eta )

            i = i + 1
            if (i % nsubLopp) == 0:
                lss = session.run(loss)
                print(lss)
                if np.isnan(lss):
                    save_path = saver.restore(session, model_path_file)
                    eta = eta / 10.0

                    print("reduzed step to ", eta)
                    if eta <= 1e-12: break
                    j = 1
                    continue

            if (i % nsubLopp) == 0:
                j = 1
                k = k - 1
                print_model_parameters(session, loss)
                lss = session.run(loss)
                if old_lss is not None:
                    if abs(lss - old_lss) < 0.00001 * abs(lss):
                        print(abs(lss - old_lss), abs(lss))
                        break
                    if lss > old_lss:
                        # mantem o loss sempre decrescente
                        save_path = saver.restore(session, model_path_file)
                        eta = eta / 5.0
                        lss = session.run(loss)
                    else:
                        save_path = saver.save(session, "/tmp/model.ckpt")
                        eta = eta * 1.05

                old_lss = lss + 0
            if k <= 0:
                k = 5
                lss = session.run(loss)
                plot_hist(uuid_named, Mm_, session, xDist_1, -lss)

        bic_md = BIC(xDist_1, XP, len(xo_), err)
        bic = session.run(bic_md)
        print("BIC =", bic)

        output_file = open("results/results_" + str(len(xo_)) + ".txt", "a")
        print("\n\n", file=output_file)
        print(uuid_named, file=output_file)
        print_model_parameters(session, loss, fileOut=output_file)
        print(" BIC =", bic, file=output_file)
        print("\n", file=output_file)
        output_file.close()
        lss = session.run(loss)
        plot_hist(uuid_named, Mm_, session, xDist_1, -lss)


def load_data(fname):
    Mx, Sx = [], []
    for l in open(fname).readlines():
        l = l.strip()
        xy = l.split(" ")
        if len(xy) >= 2:
            # print(xy)
            # print( float(xy[0]) , float(xy[1]) )
            Mx.append(float(xy[0]))
            Sx.append(float(xy[1]))
    if (len(Mx)) < 0:
        print("Unable to Load File")
    return Mx, Sx


if __name__ == "__main__":
    data = ""
    center = 0.0
    num_gaussians =2

    exec(open("./config.txt").read())
    # config = configparser.ConfigParser()
    # config.read('config.txt')
    Mm, sigmaM =  load_data(data)
    xa,xb=  np.min(Mm), np.max(Mm)
    st = np.std(Mm)
    print(xa,xb,st,np.mean(Mm))
    sx_default = float(st +0.0)
    if not os.path.exists("results/"):
        os.makedirs("results/")

    random.seed(723188)
    np.random.seed(723188)
    x1 = (np.min(Mm) +  np.mean(Mm))
    x2 = (np.max(Mm) + np.mean(Mm))
    print(x1,x2)

    ddx = float(0.5*(x2- x1))
    x_init = float(x1 /2.0)

    #xo_list = list([[center] + [np.random.random() * 0.6 + 0.04 for i in range(2)] for tries in range(100)])
    xo_list = list([[center] + [np.random.random() * ddx +x_init for i in range(num_gaussians)] for tries in range(100)])

    for xo in xo_list[5:34]:
        xo.sort()
        tf.reset_default_graph()
        compute_gaussian_fit(xo, Mm, sigmaM,sx_default)
