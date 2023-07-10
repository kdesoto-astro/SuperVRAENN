import jax.numpy as jnp

# Gaussian process params
PARAMS_INIT = jnp.array([
            0.,
            jnp.log(10.0),
            jnp.log(10.0),
            jnp.log(0.01),
            0.,
            3.
        ])

# Input prep params
NEW_LC_MIN_NPOINT = 5
NEW_LC_MAX_NPOINT = 30
PAD_SIZE = 300

# RAENN default params
NEURON_N_DEFAULT = 64
ENCODING_N_DEFAULT = 6
N_EPOCH_DEFAULT = 1000
EARLY_STOPPING_PATIENCE = 10000
BATCH_SIZE = 512

