from jax.config import config

config.update("jax_enable_x64", True)

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as random
import bridgestan
import blackjax


# cf https://blackjax-devs.github.io/blackjax/examples/howto_other_frameworks.html
# and https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html


def logp_wrapper(model):
    @jax.custom_vjp
    def logp(theta):
        return jax.pure_callback(
            lambda x: np.array(model.log_density(x)),
            np.array(1.0),
            theta,
        )

    def lopg_fwd_internal(theta):
        lp, grad = model.log_density_gradient(theta)
        return np.array(lp), grad

    def logp_fwd(theta):
        lp, grad = jax.pure_callback(
            lopg_fwd_internal, (np.array(1.0), theta), theta, vectorized=False
        )

        return lp, (grad,)

    def logp_bwd(res, g):
        return (res[0] * g,)

    logp.defvjp(logp_fwd, logp_bwd)

    return logp




def constrain_wrapper(model):
    retval = np.zeros(model.param_num(include_gq=True))

    def constrain_internal(theta, key):
        rng = model.new_rng(key[0])
        return model.param_constrain(theta, include_gq=True, rng=rng)

    def constrain(theta, key):
        return jax.pure_callback(
            constrain_internal,
            retval,
            theta,
            key
        )

    return constrain


model = bridgestan.StanModel(
    "./test_models/multi/multi_model.so", "./test_models/multi/multi.data.json"
)
M = 10
logp = logp_wrapper(model)

print(jax.grad(logp)(np.arange(0.0, M)))
print(jax.jit(logp)(np.arange(0.0, M)))
print(jax.jit(jax.vmap(logp))(np.arange(0.0, M*3).reshape(3,M)))


def other_logp(theta):
    return -0.5 * theta.dot(theta)


stepsize = 0.25
steps = 10

N = 1000

out = jnp.empty((N, M))

rng_key = random.PRNGKey(314)
nuts = blackjax.nuts(logp, 0.25, np.ones(M))
state = nuts.init(np.zeros(M))

step = jax.jit(nuts.step)
for i in range(N):
    if (i + 1) % 100 == 0:
        print(i)
    _, rng_key = jax.random.split(rng_key)
    state, info = step(rng_key, state)

    out = out.at[i].set(state.position)

print(out.mean(axis=0))
print(out.var(axis=0, ddof=1))


# vmap over param constrain

model2 = bridgestan.StanModel(
    "./test_models/bernoulli/bernoulli_model.so", "./test_models/bernoulli/bernoulli.data.json")

key = random.PRNGKey(1)
theta = np.array([0.0]) # 0.5 on the constrained scale
keys = random.split(key, 100)
constrain = constrain_wrapper(model2)
# reuse same parameter value
out = jax.vmap(constrain, (None,0))(theta, keys)
print(out)
print(out.sum(axis=0)[1] / 100) # should be approximate 0.5

thetas = np.arange(-50,50) / 10 # pretty spread out on unconstrained scale
out2 = jax.vmap(constrain, (0,0))(thetas, keys)
print(out2)
