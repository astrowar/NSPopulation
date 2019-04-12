import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np

np.random.seed(723188)



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




def model_mixture(n=2):
    mx = [tf.Variable(1.0 / float(n + 3), dtype=tf.float32, name='mix' + str(j)) for j in range(1, n)]
    sx = [tf.Variable(1.0, dtype=tf.float32, name='s' + str(j)) for j in range(1, n + 1)]
    xx = [tf.Variable(0.0 + j / n, dtype=tf.float32, name='x' + str(j)) for j in range(1, n + 1)]

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
    beta = tfd.Beta(2.0, 5.0)
    #beta =tfd.Normal(loc=0.0, scale=1.0)
    bijector = tfd.bijectors.AffineScalar(shift=xx[1], scale=sx[1])
    beta_shift = tfd.TransformedDistribution(
        distribution=beta, bijector=bijector, name="test")

    xDist = tfd.Mixture(cat=tfd.Categorical(probs=mix_cat), components=[tfd.Normal(loc=xx[0], scale=sx[0]), beta_shift])

    return beta_shift, mix_cat, xx, sx



num_gaussians = 2

xDist, mvars, xvars, svars = model_mixture(num_gaussians)


XP = tf.constant(Mm, dtype=tf.float32)

pProbs = xDist.prob(XP, name='pProbs')

xProbs = xDist.log_prob(XP, name='xProbs')
loss = -tf.reduce_mean((xProbs), name='loss')


init = tf.initialize_all_variables()
def optimize():
    with tf.Session() as session:
        session.run(init)
        # file_writer = tf.summary.FileWriter('c:\\dev\\', session.graph)

        old_lss = -9999999

        lss = session.run(pProbs)
        print(lss)
        xxrange = tf.range(0, 4, 0.03)
        yDist = xDist.prob(xxrange, name='yProbs')
        ypts = session.run(yDist)
        return list(session.run(xxrange)), list(ypts)
        # print("starting at", "x:", session.run(x1), "log(x)^2:", session.run(log_x_squared))
        # for step in range(10):
        #    session.run(train)
        #    print("step", step, "x:", session.run(x1), "log(x)^2:", session.run(log_x_squared))





xmm, ymm = optimize()


import matplotlib.pyplot as plt

# Annotate diagram
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[8, 5])
#ax1.hist(pts, bins=50, density=True, alpha=0.5, range=(0, 4), color="#707000")
ax.hist(Mm, bins=50, density=True, alpha=0.5, range=(0, 4), color="#0070FF")
ax.plot(xmm, ymm, color="crimson", lw=2, label="GMM")


ax.set_ylabel("Probability density")
ax.set_xlabel("Mass")

# Draw legend
plt.legend()
plt.show()


