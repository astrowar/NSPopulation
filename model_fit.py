import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np

np.random.seed(723188)

fdata = tf.float32



def gauss_function(x, amp, x0, sigma):
    # return tfd.Normal( x0,sigma).prob(x)
    u = tf.divide((x - x0), (2.0 * sigma))
    v = tf.divide(tf.exp(-1.0 * tf.square(u)), tf.sqrt(tf.square(sigma)))
    # v = tf.exp(-1.0 * tf.square(u))
    return amp * v

def f_gauss(x, x0, s):
    u= (x - x0)/ s
    return np.exp(-0.5* u*u )/  (s * 2.50662827463 )

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




def span_points(x, s, n=100):
    """ converte 1 ponto de amostragem em N pontos centralizados em x com dispersão s """
    if (n == 1):
        return np.array([x])
    # print(s, np.random.randn(4)/10.0 )
    return x + s * np.random.randn(n)



def get_points(X,S,  n=100):
    """ obtem os ponto com incertezas embutidas"""
    sample = []
    nn = len(X)
    for i in range(nn):
        xq = span_points(X[i], S[i], n)
        sample.append(  xq[xq >= 0] )  #adiocnao pontos positivos apenas
        #sample.append(span_points(Mm[i], sigmaM[i], n))
    return np.concatenate(sample)


def BIC(model, Xpts, num_gaussians):
    """ modelo de BIC """
    n_parameters = (num_gaussians * 3 - 1)

    npts = -1
    if isinstance(Xpts, (  np.ndarray)):
        npts = Xpts.get_shape().as_list()[0]
    else :
        npts = len(Xpts)

    #npts = Xpts.get_shape().as_list()[0]
    print(npts)
    logLikMean = tf.reduce_mean(model.log_prob(Xpts))
    return - 2 * logLikMean + n_parameters * np.log(npts)  # da wikipedia



def get_variables(pts,n=3):
    """cria as variaveis do modelo """

    xi, si , mvals  = init_gaussian_variables(pts,n)

    mx = [tf.Variable( mvals[j-1] , dtype=fdata, name='mix' + str(j)) for j in range(1, n)]
    sx = [tf.Variable( si[j-1] , dtype=fdata, name='s' + str(j)) for j in range(1, n + 1)]
    xx = [tf.Variable( xi[j-1], dtype=fdata, name='x' + str(j)) for j in range(1, n + 1)]



    #xx = [tf.constant( xi[j-1], dtype=fdata, name='x' + str(j)) for j in range(1, n + 1)]

    mix_cat = [1.0]
    if n == 2:   mix_cat = [mx[0], 1.0 - mx[0]]
    if n == 3: mix_cat = [mx[0], mx[1], 1.0 - mx[0] - mx[1]]
    if n == 4: mix_cat = [mx[0], mx[1], mx[2], 1.0 - mx[0] - mx[1] - mx[2]]
    if n == 5: mix_cat = [mx[0], mx[1], mx[2], mx[3], 1.0 - mx[0] - mx[1] - mx[2] - mx[3]]
    if n == 6: mix_cat = [mx[0], mx[1], mx[2], mx[3], mx[4], 1.0 - mx[0] - mx[1] - mx[2] - mx[3] - mx[4]]
    if n == 7: mix_cat = [mx[0], mx[1], mx[2], mx[3], mx[4], mx[5], 1.0 - mx[0] - mx[1] - mx[2] - mx[3] - mx[4] - mx[5]]
    return  mix_cat, xx, sx



def model_mixture(mix_cat, xx, sx ):
    """gera o modelo estatistico"""
    n = len(xx)
    comp_x = [tfd.Normal(loc=xx[j], scale=sx[j]) for j in range(n)]
    xDist = tfd.Mixture(cat=tfd.Categorical(probs=mix_cat), components=comp_x)
    return xDist


def init_gaussian_variables(pts , n ):
    """determina o valor inicial da variancia e centro das gaussianas por media simples"""
    xvar = np.var(pts)
    xmean = np.mean(pts)
    var_ln = (np.max(pts) - np.min(pts))
    xmin = np.min(pts)
    print(xvar,xmean)

    si = []
    xi = []
    #print([ ( 0.5+i - n/2.0 )  for i in range(n)])
    for i in range(n):

        xi.append( xmean + ( 0.5+i - n/2.0 ) *  xvar  )
        si.append( 3.5 *  xvar   )
    #si = [0.0009129944, 0.00071762624, 0.012100535, 0.16126502]
    #xi =[0.15112922, 0.14530309, 0.16224127, 0.5291017]

    vals = [0.1+f_gauss(xi[j - 1], xmean+0, xvar+0) for j in range(1, n+1)]
    vsum = np.sum(vals)
    mvals =[ x / vsum  for x in vals]

    return xi,si , mvals


def print_model_parameters(session, mvars,xvars,svars, loss ,xbic):
    if len(mvars) > 1 :
        print("M: ", [session.run(mj) for mj in mvars])
    print("X: ", [session.run(xj) for xj in xvars])
    print("S: ", [session.run(sj) for sj in svars])
    lss = session.run(loss)
    _bic = session.run(xbic)
    print("loss ", lss)
    print("BIC ", _bic)



def optimize(XP,xDist,mvars,xvars,svars,loss,train_a,xbic, clip = [] ):
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        # file_writer = tf.summary.FileWriter('c:\\dev\\', session.graph)

        #print( session.run(xDist.prob(XP) + 0.001))
        #ccc = (session.run( tf.log( xDist.prob(XP))))

        old_lss = -9999999.0
        step = -1
        while True :
            step = step + 1
            for c in clip: session.run(c)
            if (step % 5000  == 0):
                step = 1
                print_model_parameters(session,mvars,xvars,svars,loss,xbic)
                lss = session.run(loss)
                if (np.abs(lss - old_lss) < np.abs(old_lss * 0.0001)):
                    #modelo estavel
                    break
                old_lss = lss + 0
            session.run(train_a)

        print_model_parameters(session,mvars, xvars, svars, loss, xbic)

        xa,xb = np.min(XP), np.max(XP)

        xxrange = tf.range(0.0, 1.0, 0.0001)
        #xxrange = tf.range(xa, xb, 0.03)
        yDist = xDist.prob(xxrange, name='yProbs')
        ypts = session.run(yDist)
        return list(session.run(xxrange)), list(ypts)



def full_fit(X,S,num_gaussians):
    pts = get_points(X,S,50)  # 50 pontos, isso afeta o BIC
    XP = tf.constant(pts, dtype=fdata)
    #num_gaussians = 3
    mvars, xvars, svars = get_variables(pts, num_gaussians)

    xDist = model_mixture(mvars, xvars, svars)
    zProbs = xDist.prob(XP) + 1e-12
    eta = 1e-6 # learning rate
    loss = -tf.reduce_mean(tf.log(zProbs), name='loss')
    train_a = tf.train.GradientDescentOptimizer(learning_rate=eta).minimize(loss)
    xbic = BIC(xDist, X, num_gaussians)

    return  pts, XP ,mvars, xvars, svars ,xDist ,train_a, loss , xbic







