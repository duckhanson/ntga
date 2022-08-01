import os
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


def main(t: int = 64, nb_iter: int = 10, model_type: str = 'fnn', block_size: int = 128, batch_size: int = 30, dataset_name: str = 'cifar10', save_path: str ='/share/lucuslu/ntga/chlu/datasets'):
    """
    :param t: "time step used to compute poisoned data"
    :param model_type: "surrogate model backbone, either 'fnn' or 'cnn'"
    :param nb_iter: "number of iteration used to generate poisoned data"
    :param block_size: "block size of B-NTGA"
    """
    # Load data
    train_data, val_data, test_data, eps, num_classes = load_datasets(dataset_name=dataset_name, batch_size=block_size, save_path=save_path)

    # eps: "epsilon. Strength of NTGA"
    # nb_iter: "number of iteration used to generate poisoned data"
    eps_iter = (eps/nb_iter)*1.1

    # Build model
    print("Building model...")
    key = random.PRNGKey(0)
    b_std, W_std = np.sqrt(0.18), np.sqrt(1.76) # Standard deviation of initial biases and weights
    init_fn, apply_fn, kernel_fn = surrogate_fn(model_type, W_std, b_std, num_classes)
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    print("Finish Building model")

    # grads_fn: a callable that takes an input tensor and a loss function, 
    # and returns the gradient w.r.t. an input tensor.
    grads_fn = jit(grad(adv_loss, argnums=0), static_argnums=(4, 5, 7))

    # Generate Neural Tangent Generalization Attacks (NTGA)
    print("Generating NTGA....")

    x_train_adv = []
    y_train_adv = []
    for batch, ((_x_train, _y_train), (x_val, y_val)) in tqdm(enumerate(zip(train_data, val_data))):
        print(_x_train.detach().numpy().shape)
        _x_train_adv = projected_gradient_descent(model_fn=model_fn, kernel_fn=kernel_fn, grads_fn=grads_fn, 
                                                  x_train=_x_train.detach().numpy(), y_train=_y_train.detach().numpy(), x_test=x_val.detach().numpy(), y_test=y_val.detach().numpy(), 
                                                  t=t, loss='cross-entropy', eps=eps, eps_iter=eps_iter, 
                                                  nb_iter=nb_iter, clip_min=0, clip_max=1, batch_size=batch_size)

        x_train_adv.append(_x_train_adv)
        y_train_adv.append(_y_train)

        # Performance of clean and poisoned data
        _, y_pred = model_fn(kernel_fn=kernel_fn, x_train=_x_train, x_test=x_test, y_train=_y_train)
        print("Clean Acc: {:.2f}".format(accuracy(y_pred, y_test)))
        _, y_pred = model_fn(kernel_fn=kernel_fn, x_train=x_train_adv[-1], x_test=x_test, y_train=y_train_adv[-1])
        print("NTGA Robustness: {:.2f}".format(accuracy(y_pred, y_test)))

    # Save poisoned data
    x_train_adv = np.concatenate(x_train_adv)
    y_train_adv = np.concatenate(y_train_adv)

    if dataset_name == "mnist":
        x_train_adv = x_train_adv.reshape(-1, 28, 28, 1)
    elif dataset_name == "cifar10":
        x_train_adv = x_train_adv.reshape(-1, 32, 32, 3)
    elif dataset_name == "imagenet":
        x_train_adv = x_train_adv.reshape(-1, 224, 224, 3)
    else:
        raise ValueError("Please specify the image size manually.")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save('{:s}x_train_{:s}_ntga_{:s}.npy'.format(save_path, dataset_name, model_type), x_train_adv)
    np.save('{:s}y_train_{:s}_ntga_{:s}.npy'.format(save_path, dataset_name, model_type), y_train_adv)
    print("================== Successfully generate NTGA! ==================")


if __name__ == "__main__":
    main()

