import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np

np.random.seed(723188)


def gauss_function(x, amp, x0, sigma):
    # return tfd.Normal( x0,sigma).prob(x)
    u = tf.divide((x - x0), (2.0 * sigma))
    v = tf.divide(tf.exp(-1.0 * tf.square(u)), tf.sqrt(tf.square(sigma)))
    # v = tf.exp(-1.0 * tf.square(u))
    return amp * v


def logLik():
    baseDist = tf.distributions.Normal(0, 10)
    logPrior = tf.reduce_sum(baseDist.log_prob())
    return (logPrior)


Mm = [1.17, 1.57, 2.44, 1.770, 1.71, 1.00, 1.145, 1.285, 1.486, 1.073, 1.53, 1.44, 1.96, 1.545, 2.39, 1.47, 1.42, 2.56,
      2.0, 1.25, 1.34, 1.53, 1.04, 1.248, 1.365, 1.23, 1.49, 1.3332, 1.3452, 1.4398, 1.3886, 1.358, 1.354, 1.3381,
      1.2489, 1.312, 1.258, 1.3655, 1.2064, 1.38, 1.64, 1.53, 1.26, 1.57, 1.70, 1.26, 1.76, 1.27, 1.91, 1.79, 1.47,
      1.48, 1.24, 1.49, 2.08, 2.74, 1.26, 1.47, 1.34, 1.97, 1.85, 1.60, 1.0, 1.4, 1.19, 1.30, 1.205, 2.01, 1.4378, 1.58,
      1.667, 1.71]
sigmaM = [0.04, 0.04, 0.27, 0.083, 0.21, 0.10, 0.074, 0.051, 0.082, 0.358, 0.30, 0.10, 0.36, 0.465, 0.36, 0.38, 0.26,
          0.52, 0.4, 0.35, 0.37, 0.63, 0.73, 0.018, 0.018, 0.33, 0.33, 0.001, 0.001, 0.002, 0.002, 0.01, 0.01, 0.0007,
          0.0007, 0.017, 0.017, 0.002, 0.002, 0.10, 0.22, 0.08, 0.17, 0.12, 0.17, 0.14, 0.20, 0.01, 0.10, 0.10, 0.03,
          0.06, 0.11, 0.27, 0.19, 0.21, 0.39, 0.07, 0.08, 0.04, 0.15, 0.6, 0.6, 0.7, 0.29, 0.4, 0.305, 0.04, 0.0013,
          0.34, 0.021, 0.16]

pts = [1.17, 1.57, 2.44, 1.770, 1.71, 1.00, 1.145, 1.285, 1.486, 1.073, 1.53, 1.44, 1.96, 1.545, 2.39, 1.47, 1.42, 2.56,
       2.0, 1.25, 1.34, 1.53, 1.04, 1.248, 1.365, 1.23, 1.49, 1.3332, 1.3452, 1.4398, 1.3886, 1.358, 1.354, 1.3381,
       1.2489, 1.312, 1.258, 1.3655, 1.2064, 1.38, 1.64, 1.53, 1.26, 1.57, 1.70, 1.26, 1.76, 1.27, 1.91, 1.79, 1.47,
       1.48, 1.24, 1.49, 2.08, 2.74, 1.26, 1.47, 1.34, 1.97, 1.85, 1.60, 1.0, 1.4, 1.19, 1.30, 1.205, 2.01, 1.4378,
       1.58, 1.667, 1.71]


def span_points(x, s, n=100):
    if (n == 1):
        return np.array([x])
    # print(s, np.random.randn(4)/10.0 )
    return x + s * np.random.randn(n)


def get_points(n=100):
    sample = []
    nn = len(Mm)
    for i in range(nn):
        xq = span_points(Mm[i], sigmaM[i], n)
        sample.append(  xq[xq >= 0] )
        #sample.append(span_points(Mm[i], sigmaM[i], n))
    return np.concatenate(sample)


def BIC(model, Xpts, num_gaussians):
    n_parameters = (num_gaussians * 3 - 1)
    npts = Xpts.get_shape().as_list()[0]
    print(npts)
    logLikMean = tf.reduce_mean(model.log_prob(Xpts))
    return - 2 * logLikMean + n_parameters * np.log(npts)  # da wikipedia


def model_mixture(n=2):
    mx = [tf.Variable(1.0 / float(n + 3), dtype=tf.float32, name='mix' + str(j)) for j in range(1, n)]
    sx = [tf.Variable(8.0, dtype=tf.float32, name='s' + str(j)) for j in range(1, n + 1)]
    xx = [tf.Variable(1.0 + j / n, dtype=tf.float32, name='x' + str(j)) for j in range(1, n + 1)]

    mix_cat = [1.0]
    if n == 2:   mix_cat = [mx[0], 1.0 - mx[0]]
    if n == 3: mix_cat = [mx[0], mx[1], 1.0 - mx[0] - mx[1]]
    if n == 4: mix_cat = [mx[0], mx[1], mx[2], 1.0 - mx[0] - mx[1] - mx[2]]
    if n == 5: mix_cat = [mx[0], mx[1], mx[2], mx[3], 1.0 - mx[0] - mx[1] - mx[2] - mx[3]]
    if n == 6: mix_cat = [mx[0], mx[1], mx[2], mx[3], mx[4], 1.0 - mx[0] - mx[1] - mx[2] - mx[3] - mx[4]]
    if n == 7: mix_cat = [mx[0], mx[1], mx[2], mx[3], mx[4], mx[5], 1.0 - mx[0] - mx[1] - mx[2] - mx[3] - mx[4] - mx[5]]

    comp_x = [tfd.Normal(loc=xx[j], scale=sx[j]) for j in range(n)]

    # xDist = tfd.Mixture(cat=tfd.Categorical(probs=mix_cat), components=comp_x)

    # xDist = tfd.Exponential( sx[0] )

    mix_cat = [mx[0], 1.0 - mx[0]]
    beta = tfd.Beta(2.0, 2.0)
    #beta =tfd.Normal(loc=0.0, scale=1.0)
    bijector = tfd.bijectors.AffineScalar(shift=xx[1], scale=sx[1])
    beta_shift = tfd.TransformedDistribution(
        distribution=beta, bijector=bijector, name="test")

    xDist = tfd.Mixture(cat=tfd.Categorical(probs=mix_cat), components=[tfd.Normal(loc=xx[0], scale=sx[0]), beta_shift])

    return beta_shift, mix_cat, xx, sx


pts = get_points(50)
# pts = Mm

XP = tf.constant(pts, dtype=tf.float32)


# N,P = len(pts), len(pts) # number and dimensionality of observations
# Xbase = np.random.multivariate_normal(mean=np.zeros((P,)), cov=np.eye(P), size=N)
# X = tf.constant(pts, dtype=tf.float32)

def old_mixing():
    ## construct model
    ##X = tf.placeholder(dtype=tf.float32, shape=(None, P), name='X')
    # x1 = tf.Variable(0.2, dtype=tf.float32, name='x1')

    s1 = tf.Variable(2.0, dtype=tf.float32, name='s1')
    s2 = tf.Variable(0.1, dtype=tf.float32, name='s2')
    s3 = tf.Variable(0.1, dtype=tf.float32, name='s3')
    s4 = tf.Variable(0.1, dtype=tf.float32, name='s4')

    # x1 = tf.constant(1.3, dtype=tf.float32, name='x1')
    # x2 = tf.constant(1.5, dtype=tf.float32, name='x2')

    x1 = tf.Variable(1.0, dtype=tf.float32, name='x1')
    x2 = tf.Variable(1.3, dtype=tf.float32, name='x2')
    x3 = tf.Variable(1.8, dtype=tf.float32, name='x3')
    x4 = tf.Variable(2.3, dtype=tf.float32, name='x4')

    # x1 = tf.constant(1.25, dtype=tf.float32, name='x1')
    # x2 = tf.constant(1.4, dtype=tf.float32, name='x2')
    # x3 = tf.constant(1.85, dtype=tf.float32, name='x3')
    # x4 = tf.constant(2.3, dtype=tf.float32, name='x4')

    mix1 = tf.Variable(0.2, dtype=tf.float32, name='mix1')
    mix2 = tf.Variable(0.2, dtype=tf.float32, name='mix2')
    mix3 = tf.Variable(0.2, dtype=tf.float32, name='mix3')

    # mix = tf.constant(0.2, dtype=tf.float32, name='mix')
    xDist_ols = tfd.Mixture(
        cat=tfd.Categorical(probs=[mix1, mix2, 1.0 - mix1 - mix2]),
        components=[
            tfd.Normal(loc=x1, scale=s1),
            tfd.Normal(loc=x2, scale=s2),
            tfd.Normal(loc=x3, scale=s3)
            # tfd.Normal(loc=x4, scale=s4)
        ])


# xDist = tfd.Normal(x1,s1)

num_gaussians = 2
xDist, mvars, xvars, svars = model_mixture(num_gaussians)

xProbs = xDist.log_prob(XP, name='xProbs')

## prepare optimizer
eta = 1e-5  # learning rate
loss = -tf.reduce_mean((xProbs), name='loss')
train_a = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss)

# a1 = tf.Variable(2, name='a1', dtype=tf.float32)
# x1 = tf.Variable(0.0, name='x1', dtype=tf.float32)
# s1 = tf.Variable(1.0, name='s1', dtype=tf.float32)


optimizer_g = tf.train.GradientDescentOptimizer(0.01)
train_g = optimizer_g.minimize(loss)

xbic = BIC(xDist, XP, num_gaussians)

init = tf.initialize_all_variables()

clip_x1 = xvars[0].assign(tf.maximum(1.38, tf.minimum(1.42, xvars[0])))
clip_x2 = xvars[1].assign(tf.maximum(1.78, tf.minimum(1.82, xvars[1])))
clip = tf.group(clip_x1, clip_x2)

if (num_gaussians > 2):
    clip_x3 = xvars[2].assign(tf.maximum(1.23, tf.minimum(1.27, xvars[2])))
    clip = tf.group(clip_x1, clip_x2, clip_x3)


def optimize():
    with tf.Session() as session:
        session.run(init)
        # file_writer = tf.summary.FileWriter('c:\\dev\\', session.graph)

        old_lss = -9999999

        for step in range(300000):
            session.run(train_a)
            # session.run(clip)

            if (step % 1000  == 0):
                print([session.run(mj) for mj in mvars])
                print([session.run(xj) for xj in xvars])
                print([session.run(sj) for sj in svars])

                # print("step", step, "mix:", session.run(mix1), session.run(mix2), session.run(mix3))
                # print("step", step, "x:", session.run(x1), session.run(x2), session.run(x3), session.run(x4))
                lss = session.run(loss)
                _bic = session.run(xbic)

                # print("step", step, "s:", session.run(s1), session.run(s2), session.run(s3), session.run(s4), lss)  # ,,session.run((gauss_x1_bs)) ,
                print(lss)
                print("BIC ", _bic)
                if (np.abs(lss - old_lss) < np.abs(old_lss * 0.0001)):
                    break
                old_lss = lss + 0

        xxrange = tf.range(0, 4, 0.03)
        yDist = xDist.prob(xxrange, name='yProbs')
        ypts = session.run(yDist)
        return list(session.run(xxrange)), list(ypts)
        # print("starting at", "x:", session.run(x1), "log(x)^2:", session.run(log_x_squared))
        # for step in range(10):
        #    session.run(train)
        #    print("step", step, "x:", session.run(x1), "log(x)^2:", session.run(log_x_squared))


xmm, ymm = optimize()

print(xmm, ymm)

import matplotlib.pyplot as plt

# Annotate diagram
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=[8, 5])
ax1.hist(pts, bins=50, density=True, alpha=0.5, range=(0, 4), color="#707000")
ax2.hist(Mm, bins=50, density=True, alpha=0.5, range=(0, 4), color="#0070FF")
ax1.plot(xmm, ymm, color="crimson", lw=2, label="GMM")
ax2.plot(xmm, ymm, color="crimson", lw=2, label="GMM")

ax1.set_ylabel("Probability density")
ax2.set_xlabel("Mass")

# Draw legend
plt.legend()
plt.show()
