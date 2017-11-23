import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal as mv_norm
from scipy.stats import norm
import seaborn
from scipy.spatial.distance import cdist
import sklearn.gaussian_process
from sklearn.metrics.pairwise import rbf_kernel

def NonParametricRegression():
    x = np.linspace(-np.pi ,np.pi, 7)
    epsilon = np.random.normal(0.0, 0.5, x.shape[0])
    y = np.sin(x) + np.transpose(x)*epsilon

    x_star = np.random.uniform(-np.pi, np.pi, 40)
    x_star = np.sort(x_star).reshape((40, 1))

    gp = sklearn.gaussian_process.GaussianProcessRegressor()
    x = np.reshape(x,newshape=(x.shape[0],1))


    gp.fit(x, y)
    f_star, f_std = gp.predict(x_star, return_std=True)

    plt.fill_between(x_star.flatten(), f_star - f_std, f_star + f_std,
                     alpha=0.2, color='k')

    plt.plot(x_star, f_star, c='blue')
    plt.scatter(x_star, f_star, c='blue', label='predictive mean')
    plt.scatter(x,y, c='red', label='data points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Prediction using Gaussian Process')
    plt.legend()
    plt.plot()
    plt.savefig('posterior.png')



def NonParametricPrior():
    x = np.linspace(-1, 1, 201)
    W = np.array([-1.3, 0.5])
    epsilon = np.random.normal(0.0, 0.3, x.shape[0])
    y = W[0] * x + W[1] + epsilon

    x -= x.mean()
    y -= y.mean()
    X = np.stack((x,y), axis=-1)

    for l in [1, 5, 10, 100, 500]:
        sigma = 0.3
        x_axis = np.linspace(-1, 1, 500)
        for i, x_i in enumerate([X[np.random.randint(0, X.shape[0]),0]]):
            var = squaredExponantialCov(x_i, X[:,0], sigma, l)
            prior_i = mv_norm(0, var)
            y_axis= prior_i.pdf(x_axis)
            plt.plot(x_axis ,y_axis, label='l=' + str(l))

    plt.legend()
    plt.ylabel('p(f)')
    plt.xlabel('f')
    plt.title('Prior as a function of l')
    plt.savefig('priorasafunctionofl.png')



def squaredExponantialCov(x_i, x_j,sigma, l):
    d = x_i-x_j
    innerprod = np.matmul(np.transpose(d), d)

    return (sigma)*(innerprod)/(l**2)


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
    #plt.show()
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
    # plt.show()
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
    #plt.show()
    plt.close()

