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


def model_wrapper(model):
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

    def logp_bwd(res, _):
        return res

    logp.defvjp(logp_fwd, logp_bwd)

    return logp


model = bridgestan.StanModel(
    "./test_models/multi/multi_model.so", "./test_models/multi/multi.data.json"
)
M = 10
logp = model_wrapper(model)

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
