from tqdm import tqdm
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax.config import config
from neural_tangents import stax

from models.dnn_infinite import DenseGroup
from models.cnn_infinite import ConvGroup
from attacks.projected_gradient_descent import projected_gradient_descent
from utils import *
from utils_jax import *
from load_datasets import load_datasets

train_data, test_data = load_datasets()



