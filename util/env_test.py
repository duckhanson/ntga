
######### test nt & jax #########
# import neural_tangents

# from jax import random
# from neural_tangents import stax

# init_fn, apply_fn, kernel_fn = stax.serial(
#     stax.Dense(512), stax.Relu(),
#     stax.Dense(512), stax.Relu(),
#     stax.Dense(1)
# )

# key1, key2 = random.split(random.PRNGKey(1))
# x1 = random.normal(key1, (10, 100))
# x2 = random.normal(key2, (20, 100))

# kernel = kernel_fn(x1, x2, 'nngp')

# ######### test gpu ######### 
# import tensorflow as tf
# tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import jax
print(jax.default_backend())
