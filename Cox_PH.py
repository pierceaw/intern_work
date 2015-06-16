# Predicted Survival for Cox PH in lifelines
# Andrew Pierce 6/15/2015

# Requried Modules
import lifelines
from lifelines.plotting import plot_lifetimes
import numpy as np
import pandas as pd
import pylab as pl
%matplotlib inline

# Create Simulated Survival Data from Weibull Distribution
# Adapted from R code found at:
# http://www.uni-kiel.de/psychologie/rexrepos/posts/survivalCoxPH.html

# Number of Observations
N = 1000 

# Generate 40/60 Indicator for College Education
college = np.random.binomial(n = 1, p = .4, size = (N)) 

 # Generate Age variable
age = np.random.normal(loc = 42, scale = 10, size = N)
age = np.ceil(age) # Round ages up to whole numbers

# Calculate linear effect with random noise
XBeta = -1.4 * college + (.05) * age + np.random.normal(loc = 0, scale = .5, size = N)

# Generate event times using Weibull model
weib_alpha = 1.5 # Weibull Alpha parameter
weib_beta = 100 # Weibull Beta Parameter
U = np.random.uniform(low = 0, high = 1, size = N)
event_t = np.ceil((-np.log(U) * weib_beta * np.exp(-XBeta))**(1 / weib_alpha)) # get event times using weibull DGP

# Censor some observations
cutoff = 30 # Generate a censor length
cutoff = np.repeat(cutoff, N) 
duration = np.minimum(event_t,cutoff) # "Cut-off" observations over cutoff level
not_censor = event_t <= duration  # generate a boolean indicator of censoring
not_censor = not_censor.astype(int) # convert boolean to zeroes and ones

# Convert to data frame
data = pd.DataFrame({'duration': duration, 'event': not_censor, 'age': age, 'college': college})

# Plot observations with censoring
# plot_lifetimes(duration, event_observed = not_censor)

# Kaplan Meier Summary for Simulated Data
from lifelines import KaplanMeierFitter
kmf =  KaplanMeierFitter()
kmf.fit(duration, event_observed = not_censor)
kmf.survival_function_.plot()

# Cox-PH Model Regression
from lifelines import CoxPHFitter
cf = CoxPHFitter()
cf.fit(data, 'duration', event_col = 'event')
cf.print_summary()

## Get Predictions from Model ##

# 24 year old college grad
#college_24 = pd.DataFrame({'age':[24], 'college':[1]})
#cf.predict_survival_function(college_24).plot()

# 65 year old high school grad
#hs_65 = pd.DataFrame({'age':[65], 'college':[0]})
#cf.predict_survival_function(hs_65).plot()

# Predicted Survival for 24yr-old College Grad and 65yr-old HS Grad
mixed = pd.DataFrame({'age':[24, 65,42], 'college':[1,0,.4], 'index': ['24yr old College Grad','65yr old HS Grad','Average']})
mixed = mixed.set_index(['index']) # setting row names
cf.predict_survival_function(mixed).plot() # Plotting survival
pl.title('Probability of Survival at Time t')
pl.xlabel('Time t')
pl.ylabel('Probability of Survival')
"""
cf.predict_survival_function without the .plot() option will return a matrix-like
object that has the probability of survival at time t.
"""