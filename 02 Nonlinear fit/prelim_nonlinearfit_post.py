import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

# If we perform many instances of an experiment, e.g. output f(t) as a 
# function of time t, we can get a distribution of the values of f for a 
# given t. But what if we perform just one experiment?

# We perform just one experiment, with data given in 'dampedosc.dat'
data = np.loadtxt('dampedosc.dat')
plt.plot(data[:,0], data[:,1], 'o')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.show()
# Our objective is to use the data to estimate the best set of
# parameters -- a, om, phi, and b

# The fitting function is that of a damped oscillation.
# The parameters to be determined are the amplitude 'a', 
# angular speed 'om', phase 'phi' and damping factor 'b'.

# define the fitting function
def fitfunc(t, a, om, phi, b):
    return a*np.sin(om*t + phi) * np.exp(-b*t)

# In this example, we have N = 11 datapoints and M = 4 parameters.
# Let's perform a nonlinear fit using a built-in function in scipy
# and look at the resulting output and its plot
fit = curve_fit(fitfunc, data[:,0], data[:,1], p0=[1.1, 1.1, 0.1, 0.2])[0]
# Run help(curve_fit). what's p0?
print fit
plt.plot(data[:,0], data[:,1], 'o')
plt.plot(data[:,0], fitfunc(data[:,0], *fit))
plt.show()

# But let's digress a bit... 

# In a chi^2 distribution (or chi2 in scipy.stats and Eq. 1.29 in Kinzel), 
# the number of degrees of freedom is given by the number of points minus 
# the number of parameters or N - M. In our case, that's 11 - 4 = 7.

# We will use the chi^2 square distribution to give an estimate of the
# error bars for the parameter fit that we obtained.

# THIS IS IMPORTANT: In using the chi^2 distribution, we assume that 
# errors that got added is Gaussian distributed with vanishing mean
# and variances sigma_i^2 (see page 24); i.e. the errors are uncorrelated
# and follow a Gaussian distribution.

# In which case, the "best" set of parameters (parameters are denoted as 
# bold{a}) in Kinzel and which we try to look for in this exercise 
# as a0, om0, phi0, and b0 is the vector whose components minimize the
# qaudratic deviation chi^2, where chi^2 is defined as Eq. 1.28

# One can verify this in fact by generating a bunch of datasets 
# Exercise: Using the code from gendata_book_post.py, generate at
# least 100 datasets, calculate chi^2 using Eq. 1.28 and show that
# the distribution of chi^2(bold{a0}) is given by Eq. 1.29

# The chi^2 distribution PDF (Eq. 1.29) is available from scipy.stats
chi2.pdf(np.linspace(0,20),7)
# Use the help function to figure out the parameters: help(chi2.pdf)
# Tip: check out the other functions available help(chi2)
