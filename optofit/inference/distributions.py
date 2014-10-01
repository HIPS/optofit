import numpy as np
import scipy
from scipy.stats import gamma, invgamma

class Distribution(object):
    """
    Abstract base class for distributions
    """
    def __init__(self):
        # TODO Save shape?

        pass

    def sample(self, Np=1):
        """
        Sample from the proposald distribution given the previous state.
        """
        return None

    def logp(self, x):
        """ Compute the log probability of the state given the previous state
        """
        return -np.Inf

class ProductDistribution(Distribution):
    def __init__(self, distns):
        self.distns = distns

    def sample(self, Np=1):
        return np.array([d.sample(Np) for d in self.distns])

    def logp(self, x):
        return np.array([d.logp(xi) for xi,d in zip(x,self.distns)]).sum()

    def grad_logp(self, x):
        return np.array([d.grad_logp(xi) for xi,d in zip(x,self.distns)])

    def logp_logx(self, logx):
        return np.array([d.logp_logx(logxi) for logxi,d in zip(logx,self.distns)]).sum()

    def grad_logp_logx_wrt_x(self, x):
        return np.array([d.grad_logp_logx_wrt_x(xi) for xi,d in zip(x,self.distns)])


class GaussianDistribution(Distribution):
    """
    Simple Gaussian proposal distribution.
    """
    def __init__(self, D, mu, cov):
        """
        Initialize the Gaussian proposal with a covariance
        """
        self.D = D

        assert mu.shape == (D,1),  "ERROR: Mean mu is not the right size"
        self.mu = mu

        assert cov.shape == (D,D), "ERROR: Covariance matrix is not the right size"
        self.cov = cov
        self.chol, _ = scipy.linalg.cho_factor(cov+np.diag(np.random.rand(D)*10**-16),
                                               lower=True)

        # Round to remove noise
        self.chol[np.abs(self.chol)<1e-8] = 0.0

        # Precompute the normalization factor
        sgn, logdet = np.linalg.slogdet(self.cov)
        self.log_Z = -0.5 * np.log(2.0*np.pi) * sgn * logdet

    def sample(self, Np=1):
        """ Sample the next state given the previous
        """
        return self.mu + np.dot(self.chol, np.random.randn(self.D,Np))

    def logp(self, x):
        """ Compute the log probability of the state given the previous state
        """
        xc = x-self.mu
        tmp = scipy.linalg.cho_solve((self.chol,True), xc)
        lp = self.log_Z - 0.5 * (xc*tmp).sum(0)
        # lp =  self.log_Z - 0.5 * np.diag(np.dot((x-self.mu).T,
        #                                  np.linalg.solve(self.cov, (x-self.mu))))

        return np.squeeze(lp)


class StaticTruncatedGaussianDistribution(Distribution):
    """
    Truncated Gaussian proposal distribution.
    """
    def __init__(self, D, mu, sigma, lb, ub):
        """
        Initialize the Gaussian proposal with a covariance
        """
        self.D = D

        assert mu.size == D,  "ERROR: Mean mu is not the right size"
        self.mu = mu.reshape(D)

        assert sigma.size == D
        self.sigma = sigma.reshape(D)

        self.lb = lb.reshape(D)
        self.ub = ub.reshape(D)

        self.trunca = (self.lb - self.mu) / self.sigma
        self.truncb = (self.ub - self.mu) / self.sigma

        from scipy.stats import truncnorm
        self.truncnorms = [truncnorm(self.trunca[d], self.truncb[d])
                           for d in np.arange(D)]

    def sample(self, Np=1):
        """ Sample the next state given the previous
        """
        rvs =  np.array([self.mu[d] + self.sigma[d]*self.truncnorms[d].rvs(size=Np) for d in np.arange(self.D)])
        return rvs

    def logp(self, x):
        """ Compute the log probability of the state given the previous state
        """
        D,Np = x.shape
        lp = np.zeros(Np)
        for d in np.arange(D):
            lp += self.truncnorms[d].logpdf((x[d,:]-self.mu[d])/self.sigma[d])
        return lp


def fixed_ppf(cdf, loc=0, scale=0):
    from scipy.stats import norm

    def indexer(obj):
        if obj.shape == ():
            return lambda row, col: obj
        else:
            if obj.shape[1] == 1:
                return lambda row, col: obj[row]
            else:
                return lambda row, col: obj[row, col]

    ans = np.zeros(cdf.shape)
    mean = indexer(loc)
    sig  = indexer(scale)

    for row in range(len(cdf)):
        for col in range(len(cdf[0])):
            ans[row, col] = norm.ppf(cdf[row, col], loc=mean(row, col), scale=sig(row, col))

    return np.matrix(ans)

def check_bounds(val, lb, ub):
    try:
        assert np.all(np.isfinite(val))
        assert np.all(val>=lb), "ERROR: Sampled below lb"
        assert np.all(val<=ub), "ERROR: Sampled above ub"    
    except:
        import pdb; pdb.set_trace()
        wrong = np.logical_not(np.logical_and(np.logical_and(np.isfinite(val), val >= lb), val <= ub))
        # print np.sum(wrong.astype(np.int32))
        # print np.array(lb[wrong])
        # print np.array(ub[wrong])
        # print np.array(val[wrong])

class TruncatedGaussianDistribution(Distribution):
    """
    Truncated Gaussian proposal distribution with data-dependent
    parameters.
    """
    def __init__(self):
        """
        Initialize the Gaussian proposal with a covariance
        """
        pass

    @staticmethod
    def normal_cdf(x, mu, sigma):
        z = (x-mu)/sigma
        return 0.5 * scipy.special.erfc(-z / np.sqrt(2))

    def sample(self, mu=0, sigma=1, lb=-np.Inf, ub=np.Inf):
        """ Sample a truncated normal with the specified params
        """
        if np.allclose(sigma, 0.0):
            return mu

        cdflb = self.normal_cdf(lb, mu, sigma)
        cdfub = self.normal_cdf(ub, mu, sigma)

        # Sample uniformly from the CDF
        cdfsamples = cdflb + np.random.rand(*mu.shape)*(cdfub-cdflb)
        zs = -np.sqrt(2)*scipy.special.erfcinv(2*cdfsamples)
        rvs = sigma * zs + mu

        # Test
        pint = cdfub - cdflb

        # logp = -np.Inf * (x < lb) + -np.Inf * (x > ub)
        # logp = norm.logpdf(x, loc=mu, scale=sigma) - np.log(pint)
        # logp[x<lb] = -np.Inf
        # logp[x>ub] = -np.Inf

        # p(x) = \frac{1}{\sqrt{2*pi*sigma^2}}  \exp{-\frac{1}{\sqrt{2} \sigma^2} (x-mu)^2}
        # logp = -0.5*np.log(2*np.pi) -np.log(sigma) - 0.5/sigma**2 * (rvs-mu)**2
        # logp -= np.log(pint)
        # logp[rvs<lb] = -np.Inf
        # logp[rvs>ub] = -np.Inf


        # check_bounds(rvs, lb, ub)
        return rvs

    def logp(self, x, mu=0, sigma=1, lb=-np.Inf, ub=np.Inf):
        """ Compute the log probability of the state given the previous state
        """
        if np.allclose(sigma, 0.0):
            if np.allclose(x, mu):
                return np.Inf
            else:
                return -np.Inf

        # Compute the probability mass in the interval
        # from scipy.stats import norm
        # cdflb = norm.cdf(lb, loc=mu, scale=sigma)
        # cdfub = norm.cdf(ub, loc=mu, scale=sigma)
        cdflb = self.normal_cdf(lb, mu, sigma)
        cdfub = self.normal_cdf(ub, mu, sigma)
        # print "WARNING DEBUG TRUNC NORMAL"
        # cdflb = 0
        # cdfub = 1
        pint = cdfub - cdflb

        # logp = -np.Inf * (x < lb) + -np.Inf * (x > ub)
        # logp = norm.logpdf(x, loc=mu, scale=sigma) - np.log(pint)
        # logp[x<lb] = -np.Inf
        # logp[x>ub] = -np.Inf

        # p(x) = \frac{1}{\sqrt{2*pi*sigma^2}}  \exp{-\frac{1}{\sqrt{2} \sigma^2} (x-mu)^2}
        logp = -0.5*np.log(2*np.pi) -np.log(sigma) - 0.5/sigma**2 * (x-mu)**2
        logp -= np.log(pint)
        logp[x<lb] = -np.Inf
        logp[x>ub] = -np.Inf


        """
        assert np.all(np.isfinite(x)), np.sum(np.isfinite(x).astype(np.int32))
        assert np.all(x>=lb), "ERROR: Querying below lb"
        assert np.all(x<=ub), "ERROR: Querying above ub"
        """
        # check_bounds(x, lb, ub)
        return logp


class SphericalGaussianDistribution(GaussianDistribution):
    """
    Special case of the Gaussian proposal with spherical covariance
    """
    def __init__(self, D, mu=None, sig=1):
        if mu is None:
            mu = np.zeros((D,1))
        super(SphericalGaussianDistribution, self).__init__(D, mu, sig**2 * np.eye(D))

class DiagonalGaussianDistribution(GaussianDistribution):
    """
    Special case of the Gaussian proposal with spherical covariance
    """
    def __init__(self, D, mu=None, sig=None):
        if mu is None:
            mu = np.zeros((D,1))
        elif np.isscalar(mu):
            mu *= np.ones((D,1))
        elif isinstance(mu, np.ndarray) and mu.size == D:
            mu = mu.reshape((D,1))
        else:
            raise Exception('Invalid shape for mu!')

        # Check covariance shape
        if sig is None:
            sig = np.eye(D)
        elif np.isscalar(sig):
            sig = sig**2 * np.eye(D)
        elif isinstance(sig, np.ndarray) and sig.size == D:
            sig = np.diag(sig.ravel() ** 2)
        else:
            raise Exception('Diagonal Gaussian must be given vector of variances')

        super(DiagonalGaussianDistribution, self).__init__(D, mu, sig)


class GammaDistribution(Distribution):
    """
    Simple Gamma proposal distribution.
    """
    def __init__(self, a, b):
        """
        Initialize a gamma distribution
        """
        self.a = a
        self.b = b
        self._gamma = gamma(a, scale=1.0/b)

    @property
    def mean(self):
        return self.a/self.b

    @property
    def std(self):
        return np.sqrt(self.a/self.b**2)

    def sample(self, Np=1):
        """ Sample the next state given the previous
        """
        return self._gamma.rvs(Np)

    def logp(self, x):
        """ Compute the log probability of the state given the previous state
        """
        # return self._gamma.logpdf(x)
        return (self.a-1.0)*np.log(x) - self.b*x

    def grad_logp(self, x):
        """
        Gradient of log prob wrt x
        lp \propto (a-1)*log(x) - b*x
        dlp/dx = (a-1)/x - b
        """
        return (self.a - 1.0)/x - self.b

    def logp_logx(self, logx):
        x = np.exp(logx)
        return self.a * np.log(x) - self.b*x

    def grad_logp_logx_wrt_x(self, x):
        return self.a/x - self.b

class InverseGammaDistribution(Distribution):
    """
    Simple Inverse Gamma proposal distribution
    """
    def __init__(self, a, b):
        self._invgamma = invgamma(a, scale=b)
        self._a = a
        self._b = b

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def sample(self, Np=1):
        """ Sample the next state given the previous
        """
        return self._invgamma.rvs(Np)

    def logp(self, x):
        """ Compute the log probability of the state given the previous state
        """
        return self._invgamma.logpdf(x)

class DeltaFunction(Distribution):
    """
    A degenerate distribution for fixed parameters
    """
    def __init__(self, v):
        """
        Initialize a delta function at v
        """
        self._v = v

    def sample(self, Np=1):
        """ Sample the next state given the previous
        """
        return self._v * np.ones(Np)

    def logp(self, x):
        """ Compute the log probability of the state given the previous state
        """
        return -np.Inf * np.not_equal(x, self._v)

class NormalInverseGammaDistribution(Distribution):
    def __init__(self, D, mu, lninv, a, b):
        self.D = D
        self.mu = mu
        self.lninv = lninv
        self.a = a
        self.b = b
        self.invgamma = InverseGammaDistribution(a, b)

    def sample(self, Np):
        sigma_sq = self.invgamma.sample(Np)
        mu       = np.zeros(Np, self.D, 1)
        for s in sigma_sq:
            mu[s] = GaussianDistribution(self.D, self.mu, cov)
        
def test_truncated_gaussian():
    # First test no truncation
    lb = -np.Inf
    ub = np.Inf
    tg = TruncatedGaussianDistribution()
    N = 10000
    samples = tg.sample(mu=np.zeros((N,)), sigma=np.ones((N,)), lb=lb, ub=ub)

    from scipy.stats import norm
    import matplotlib.pyplot as plt
    _,bins,_ = plt.hist(samples,50, normed=True, alpha=0.5)
    plt.plot(bins, norm.pdf(bins))
    plt.show()

    # First test no truncation
    lb = 0
    ub = np.Inf
    tg = TruncatedGaussianDistribution()
    N = 10000
    samples = tg.sample(mu=np.zeros((N,)), sigma=np.ones((N,)), lb=lb, ub=ub)

    from scipy.stats import norm
    import matplotlib.pyplot as plt
    _,bins,_ = plt.hist(samples,50, normed=True, alpha=0.5)
    plt.plot(bins, 2*norm.pdf(bins))
    plt.show()

if __name__ == "__main__":
    test_truncated_gaussian()