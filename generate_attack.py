from tqdm import tqdm
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.config import config
from neural_tangents import stax

from attacks.projected_gradient_descent import projected_gradient_descent
from utils import *
from utils_jax import *
from load_datasets import load_datasets
from utils_generate_attack import surrogate_fn, model_fn, adv_loss


if __name__ == "__main__":
    train_data, test_data, eps, num_classes = load_datasets()

    nb_iter = 10 # "number of iteration used to generate poisoned data"
    eps_iter = (eps/nb_iter)*1.1

    model_type = 'fnn' # "surrogate function"
    # Build model
    print("Building model...")
    key = random.PRNGKey(0)
    b_std, W_std = np.sqrt(0.18), np.sqrt(1.76) # Standard deviation of initial biases and weights
    init_fn, apply_fn, kernel_fn = surrogate_fn(model_type, W_std, b_std, num_classes)
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    print("Finish Building model")


