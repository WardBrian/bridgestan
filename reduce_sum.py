import os
os.environ["STAN_NUM_THREADS"] = "1"

import bridgestan
import numpy as np

# logistic0 = bridgestan.StanModel("logistic0.stan", "redcards.json", make_args=["STAN_THREADS=true"])
logistic1 = bridgestan.StanModel("logistic1.stan", "redcards.json", make_args=["STAN_THREADS=true"])


inits_unc = np.random.uniform(low=-2, high=2, size=logistic1.param_unc_num())

def zero():
    lp, grad = logistic0.log_density_gradient(inits_unc)

def one():
    lp, grad = logistic1.log_density_gradient(inits_unc)

import timeit

# timeit.timeit(zero, number=100)
# print("Running model 0")
# time0 = timeit.timeit(zero, number=1000)
# print("Running model 1 (no threads)")
# time1 = timeit.timeit(one, number=1000)

logistic1._set_threadpool_size(4, None)
print("Running model 1 (threads)")
logistic1._set_threadpool_size(1, None)
time2 = timeit.timeit(one, number=1000)
print( time2)
