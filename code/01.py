import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal as mv_norm
import seaborn
import scipy.optimize as opt
from itertools import product

def BayesianLinearRegression(s, plotprior=False):
    x = np.linspace(-1, 1, 201)
    W = np.array([-1.3, 0.5])
    epsilon = np.random.normal(0.0, 0.3, x.shape[0])
    y = W[0]*x + W[1] + epsilon

    precision = 1/0.3
    m_0 = [0, 0]
    s_0 = (1/precision)*np.identity(2)
    prior = mv_norm(m_0, s_0)

    if plotprior:
        plotPrior(prior)

    size = s
    datapoint, idx = samplerandomdatapoints(size, x, y)
    posterior, possiblew, W0, W1 = computeposteriorOverDataPoints(prior, datapoint)
    plotFunctions(size, possiblew, posterior, datapoint)
    plotPosterior(size, posterior, W0, W1, datapoint)

def samplerandomdatapoints(s, x, y):
    idx = np.random.randint(0, x.shape[0],s)
    datapoint = np.stack((x[idx], y[idx]), axis=-1)

    return datapoint, idx

def computeposteriorOverDataPoints(prior, datapoints):

    sigma = np.cov(datapoints[:,0], datapoints[:,1])[0,0]

    possiblew = np.linspace(-3, 3, 100)
    W0, W1 = np.meshgrid(possiblew, possiblew)
    possiblew = np.stack((W0.flatten(), W1.flatten()), axis=-1)
    priorpdf = prior.pdf(possiblew)

    likelihood = 1
    for i in range(0, datapoints.shape[0]):
        likelihood_i = (1/np.sqrt(2*np.pi*(sigma**2)))
        likelihood_i = np.multiply(likelihood_i,  np.exp(np.square(datapoints[i,1]-(datapoints[i,0]*possiblew[:,0]+possiblew[:,1]))/(2*(sigma**2))))
        likelihood = np.multiply(likelihood, likelihood_i)

    likelihood /= datapoints.shape[0]

    posterior = priorpdf*likelihood

    return posterior, possiblew, W0, W1

def plotFunctions(s, possiblew, posterior, datapoint):

    indices = np.argsort(posterior)[::1]

    for i in range(0, 5):
        idx = indices[i]
        x_axis = np.linspace(-1, 1, 201)
        y_axis = x_axis*possiblew[idx, 0] + possiblew[idx, 1]
        plt.plot(x_axis, y_axis, label='$y=' + str(possiblew[idx, 0]) + 'x + ' + str(possiblew[idx,1]) + '$')
    plt.scatter(datapoint[:,0], datapoint[:,1])
    plt.legend()
    plt.title('Functions described by posterior for '+str(s)+' data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('function-'+str(s)+'datapoint.png')
    plt.close()

def plotPosterior(s, posterior, W0, W1, datapoint):

    posterior = np.reshape(posterior, (100, 100))
    cs = plt.contourf(W0,W1,posterior)
    plt.scatter(datapoint[:,0], datapoint[:,1], label='randomly selected datapoint')
    plt.legend()
    cbar = plt.colorbar(cs)
    plt.grid(False)
    plt.xlabel(r'$w_0$')
    plt.ylabel(r'$w_1$')
    plt.title('Posterior over $W$ using '+str(s)+' random datapoints')
    plt.savefig('posterior'+str(s)+'.png', facecolor='white', edgecolor='white')
    plt.close()

def plotPrior(prior):

    x = np.linspace(-2,2,100)
    xx, yy = np.meshgrid(x, x)
    values = np.stack((xx,yy), axis=-1)
    proba = prior.pdf(values)
    proba = (proba-proba.min())/(proba.max()-proba.min())
    cs = plt.contourf(xx, yy, proba)
    cbar = plt.colorbar(cs)
    plt.xlabel(r'$w_0$')
    plt.ylabel(r'$w_1$')
    plt.title('Prior over $W$')
    plt.savefig('prior.png')
    plt.close()


def plotPrior(l):
    X = np.linspace(-4.0,4.0,2000) [:,None]
    mu = np.zeros(shape=(2000))
    K = np.exp(-euclidean(X, X)/(l*l))
    Z = np.random.multivariate_normal(mu,K,20)
    for i in range(20):
        plt.plot(X[:],Z[i,:])
    plt.title('Gaussian Prior for l = '+str(l))
    plt.savefig('prior' + str(l) + '.png')

def kernelFunction(X, Y, l=1.):
    return np.exp(-euclidean(X, Y)/(l*l))

def PlotPosterior():
    X = np.linspace(-np.pi, np.pi, 7)
    epsilon = np.random.normal(0, 0.5, 7)
    Y = np.sin(X) + epsilon
    x = np.linspace(-2 * np.pi, 2 * np.pi, 800)[:,None]
    list = [1.8, 2.5]
    for l in list:
        X = X[:, None]
        k = kernelFunction(x, X, l)
        C = np.linalg.inv(kernelFunction(X, X, l))
        t = Y[:, None]
        mu = np.dot(np.dot(k, C), t)
        c = kernelFunction(x, x, l)
        sigma = c - np.dot(np.dot(k, C), np.transpose(k))

        # plot observations
        plt.plot(X, Y, 'ro', label='datapoints')
        plt.plot(x, np.sin(x), color='green', label='truth')
        mu = np.reshape(mu, (800,))
        plt.plot(x, mu, color='blue', label='regression')
        upper = mu + 2 * np.sqrt(sigma.diagonal())
        lower = mu - 2 * np.sqrt(sigma.diagonal())
        ax = plt.gca()
        ax.fill_between(x, upper, lower, facecolor='k', interpolate=True, alpha=0.3)
        plt.title('Gaussian Process Regression with l = ' + str(l))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig('regl=' + str(l) + '.png')

def plotSamplePosterior():
    X = np.linspace(-np.pi, np.pi, 7)
    epsilon = np.random.normal(0, 0.5, 7)
    Y = np.sin(X) + epsilon
    l=1.8
    x = np.linspace(-2*np.pi, 2*np.pi, 800)[:, None]
    X = X[:, None]
    k = kernelFunction(x, X, l)
    C = np.linalg.inv(kernelFunction(X, X, l))
    t = Y[:, None]
    mu = np.dot(np.dot(k, C), t)
    c = kernelFunction(x, x, l)
    sigma = c - np.dot(np.dot(k, C), np.transpose(k))
    mu = np.reshape(mu,(800,))
    x = x[:,None]
    Z = np.random.multivariate_normal(mu,np.nan_to_num(sigma),20)
    plt.plot(X,Y,'ro')
    for i in range(20):
        plt.plot(x[:],Z[i,:])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sample of the Posterior with l=1.8')
    plt.savefig('samplePos1.png')


def Optimization():
    A = np.random.randn(20)
    A = A.reshape((10, 2))

    x = np.linspace(0, 4 * np.pi, 100)
    X = np.zeros((100, 2))
    X[:, 0] = np.multiply(x, np.cos(x))
    X[:, 1] = np.multiply(x, np.sin(x))

    Y = np.dot(X, np.transpose(A))
    noise = np.random.multivariate_normal(np.zeros(10), 0.1 * np.eye(10), 100)

    plt.scatter(X[:, 0], X[:, 1])
    plt.title('True Signal')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.savefig('question20/truth.png')
    plt.close()
    Y = Y+noise

    A = 20 * np.random.randn(20)
    A = np.reshape(A, (20,))


    def functionf(W):
        sigma = 2
        W = np.reshape(W, (10, 2))
        H = np.dot(W, np.transpose(W))
        I = sigma * np.eye(10)
        inv = np.linalg.inv(H + I)
        A = 50 * np.log(np.linalg.det(H + I))
        B = 0.5 * np.trace(np.dot(inv, np.dot(np.transpose(Y), Y)))
        return A + B + 0.5 * 10 * 100 * np.log(2 * np.pi)

    def derivativef(W):
        sigma = 2
        W = np.reshape(W, (10, 2))
        inv = np.linalg.inv((np.dot(W, np.transpose(W)) + sigma * np.eye(10)))

        values = np.empty(W.shape)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                J = np.zeros(np.shape(W))
                J[i, j] = 1
                dWW = np.dot(J, np.transpose(W)) + np.dot(W, np.transpose(J))
                A = 100 * np.trace(np.dot(inv, dWW))
                B1 = np.dot(np.dot(-inv, dWW), inv)
                B = np.trace(np.dot(np.dot(np.transpose(Y), Y), B1))
                values[i, j] = 0.5 * A + 0.5 * B
        val = np.reshape(values, (20,))
        return val

    Wstar = opt.fmin_cg(functionf, A, fprime=derivativef)
    W = np.reshape(Wstar, (10, 2))
    inv = np.linalg.pinv(np.dot(np.transpose(W), W))
    X = np.dot(Y, np.dot(W, np.dot(np.transpose(W), W)))

    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Recovered Signal')

    plt.savefig('recovered.png')


def SampleFromPrior(modelNumber, samples):
    sigma = 1000
    cov = sigma*np.eye(modelNumber)
    mean = np.ones(modelNumber) * 5
    return np.random.multivariate_normal(mean, cov, samples)

def CalculateEvidence(dataset, modelNumber, samples):
    p=0
    for i in range(len(samples)):
        if modelNumber == 0:
            return 1 / 512
        p1 = 1
        for i in range(3):
            for j in range(3):
                if modelNumber == 1:
                    p1 = p1 * 1 / (1 + np.exp(-dataset[i, j] * samples[i][0] * (i - 1)))
                if modelNumber == 2:
                    p1 = p1 * 1 / (1 + np.exp(-dataset[i, j] * (samples[i][0] * (i - 1) + samples[i][1] * (j - 1))))
                if modelNumber == 3:
                    p1 = p1 * 1 / (1 + np.exp(-dataset[i, j] * (samples[i][0] * (i - 1) + samples[i][1] * (j - 1) + samples[i][2])))
        p = p + p1

    return p/len(samples)

def Evidence():
    samples1 = SampleFromPrior(1, 10 ** 4)
    samples2 = SampleFromPrior(2, 10 ** 4)
    samples3 = SampleFromPrior(3, 10 ** 4)
    combinations = list(product([-1, 1], repeat=9))
    sets = []
    for l in combinations:
        arr = np.asarray(l)
        grid = np.reshape(arr, (3, 3))
        sets.append(grid)
    l = sets

    evidence = np.zeros([4, 512])

    for i in range(4):
        print(i)
        for j in range(512):
            if i == 0:
                evidence[i][j] = CalculateEvidence(l[j], i, samples1)
            if i == 1:
                evidence[i][j] = CalculateEvidence(l[j], i, samples1)
            if i == 2:
                evidence[i][j] = CalculateEvidence(l[j], i, samples2)
            if i == 3:
                evidence[i][j] = CalculateEvidence(l[j], i, samples3)

    max = np.argmax(evidence, axis=1)
    min = np.argmin(evidence, axis=1)
    sum = np.sum(evidence, axis=1)
    print(max, min, sum)

    dist = np.zeros([evidence.shape[0], evidence.shape[0]])
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            dist[i, j] = evidence[i] - evidence[j]
            if i == j:
                dist[i, j] = pow(10, 4)

    index = [];
    D = np.arange(evidence.shape[0]).tolist()
    ind = evidence.argmin()
    index.append(ind)
    D.remove(ind)

    while D:
        N = []
        for i in range(len(D)):
            ind = dist[D[i], D].argmin()
            if D[ind] == index[-1]:
                N.append(D[ind])
        if not N:
            index.append(D[dist[index[-1], D].argmin()])
        else:
            index.append(N[dist[index[-1], N].argmin()])
        D.remove(index[-1])

    plt.plot(evidence[0, index], label="P($\mathcal{D}$ | ${M}_0$)")
    plt.plot(evidence[1, index], label="P($\mathcal{D}$ | ${M}_1$)")
    plt.plot(evidence[2, index], label="P($\mathcal{D}$ | ${M}_2$)")
    plt.plot(evidence[3, index], label="P($\mathcal{D}$ | ${M}_3$)")
    plt.legend()
    plt.show()
