import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
# If this causes an error, you can comment it out.
#import seaborn; seaborn.set()

def make_data(intercept, slope, N=20, dy=2, rseed=42):
    rand = np.random.RandomState(rseed)
    x = 100 * rand.rand(20)
    y = intercept + slope * x
    y += dy * rand.randn(20)
    return x, y, dy * np.ones_like(x)

#def ell_data()

theta_true = (2, 0.5)
x, y = angle, make_data(angle, theta[0], theta[1], theta[2]) 

#plt.errorbar(x, y, dy, fmt='o');

def model(theta, x):
    # the `theta` argument is a list of parameter values, e.g., theta = [m, b] for a line
    return theta[0] + theta[1] * x

def ln_likelihood(theta, x, y, dy):
    # we will pass the parameters (theta) to the model function
    # the other arguments are the data
    return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2)+ ((y - model(theta, x)) / dy) ** 2)

def ln_prior(theta):
    # flat prior: log(1) = 0
    return 0
def ln_posterior(theta, x, y, dy):
    return ln_prior(theta) + ln_likelihood(theta, x, y, dy)


def run_mcmc(ln_posterior, nsteps, ndim, theta0, stepsize, args=()):
    """
    Run a Markov Chain Monte Carlo
    
    Parameters
    ----------
    ln_posterior: callable
        our function to compute the posterior
    nsteps: int
        the number of steps in the chain
    theta0: list
        the starting guess for theta
    stepsize: float
        a parameter controlling the size of the random step
        e.g. it could be the width of the Gaussian distribution
    args: tuple (optional)
        additional arguments passed to ln_posterior
    """
    # Create the array of size (nsteps, ndims) to hold the chain
    # Initialize the first row of this with theta0
    chain = np.zeros((nsteps, ndim))
    chain[0] = theta0
    
    # Create the array of size nsteps to hold the log-likelihoods for each point
    # Initialize the first entry of this with the log likelihood at theta0
    log_likes = np.zeros(nsteps)
    log_likes[0] = ln_posterior(chain[0], *args)
    
    # Loop for nsteps
    for i in range(1, nsteps):
        # Randomly draw a new theta from the proposal distribution.
        # for example, you can do a normally-distributed step by utilizing
        # the np.random.randn() function
        theta_new = chain[i - 1] + stepsize * np.random.randn(ndim)
        
        # Calculate the probability for the new state
        log_like_new = ln_likelihood(theta_new, *args)
        
        # Compare it to the probability of the old state
        # Using the acceptance probability function
        # (remember that youve computed the log probability, not the probability!)
        log_p_accept = log_like_new - log_likes[i - 1]
        
        # Chose a random number r between 0 and 1 to compare with p_accept
        r = np.random.rand()
        
        # If p_accept>1 or p_accept>r, accept the step
        # Else, do not accept the step
        if log_p_accept > np.log(r):
            chain[i] = theta_new
            log_likes[i] = log_like_new
        else:
            chain[i] = chain[i - 1]
            log_likes[i] = log_likes[i - 1]
            
    return chain


chain = run_mcmc(ln_posterior, 10000, 3, (0.5, 1, 0), 0.01, (x, y, dy))
fig, ax = plt.subplots(3)
ax[0].plot(chain[:, 0])
ax[1].plot(chain[:, 1]);
ax[2].plot(chain[:, 2]);


# Now that weve burned-in, lets get a fresh chain
chain = run_mcmc(ln_posterior, 500000, 3, chain[-1], 0.1, (x, y, dy))

fig, ax = plt.subplots(3)
ax[0].plot(chain[:, 0])
ax[1].plot(chain[:, 1]);
ax[2].plot(chain[:, 2]);

fig, ax = plt.subplots(3)
ax[0].hist(chain[:, 0], bins=250, alpha=0.5, density=True)
ax[1].hist(chain[:, 1], bins=250, alpha=0.5, density=True);
ax[2].hist(chain[:, 2], bins=250, alpha=0.5, density=True);                              



# plt.hist2d(chain[:, 0], chain[:, 1], bins=[80,30],
#            cmap='Blues', density=True)
# plt.xlabel('intercept')
# plt.ylabel('slope')
# plt.grid(False);
# plt.show();

# theta_best = chain.mean(0)
# theta_std = chain.std(0)

# print("true intercept:", theta_true[0])
# print("true slope:", theta_true[1])
# print()
# print("intercept = {0:.1f} +/- {1:.1f}".format(theta_best[0], theta_std[0]))
# print("slope = {0:.2f} +/- {1:.2f}".format(theta_best[1], theta_std[1]))


