import neural_tangents as nt
from neural_tangents import stax
from models.dnn_infinite import DenseGroup
from models.cnn_infinite import ConvGroup

def surrogate_fn(model_type, W_std, b_std, num_classes):
    """
    :param model_type: string. `fnn` or `cnn`.
    :param W_std: float. Standard deviation of weights at initialization.
    :param b_std: float. Standard deviation of biases at initialization.
    :param num_classes: int. Number of classes in the classification task.
    :return: triple of callable functions (init_fn, apply_fn, kernel_fn).
            In Neural Tangents, a network is defined by a triple of functions (init_fn, apply_fn, kernel_fn). 
            init_fn: a function which initializes the trainable parameters.
            apply_fn: a function which computes the outputs of the network.
            kernel_fn: a kernel function of the infinite network (GP) of the given architecture 
                    which computes the kernel matrix
    """
    if model_type == "fnn":
        init_fn, apply_fn, kernel_fn = stax.serial(DenseGroup(5, 512, W_std, b_std))
    elif model_type == "cnn":
        if args.dataset == 'imagenet':
            init_fn, apply_fn, kernel_fn = stax.serial(ConvGroup(2, 64, (3, 3), W_std, b_std),
                                                       stax.Flatten(),
                                                       stax.Dense(384, W_std, b_std),
                                                       stax.Dense(192, W_std, b_std),
                                                       stax.Dense(num_classes, W_std, b_std))
        else:
            init_fn, apply_fn, kernel_fn = stax.serial(ConvGroup(2, 64, (2, 2), W_std, b_std),
                                                       stax.Flatten(),
                                                       stax.Dense(384, W_std, b_std),
                                                       stax.Dense(192, W_std, b_std),
                                                       stax.Dense(num_classes, W_std, b_std))
    else:
        raise ValueError
    return init_fn, apply_fn, kernel_fn

def model_fn(kernel_fn, x_train=None, x_test=None, fx_train_0=0., fx_test_0=0., t=None, y_train=None, diag_reg=1e-4):
    """
    :param kernel_fn: a callable that takes an input tensor and returns the kernel matrix.
    :param x_train: input tensor (training data).
    :param x_test: input tensor (test data; used for evaluation).
    :param y_train: Tensor with one-hot true labels of training data.
    :param fx_train_0 = output of the network at `t == 0` on the training set. `fx_train_0=None`
            means to not compute predictions on the training set. fx_train_0=0. for infinite width.
    :param fx_test_0 = output of the network at `t == 0` on the test set. `fx_test_0=None`
            means to not compute predictions on the test set. fx_test_0=0. for infinite width.
            For more details, please refer to equations (10) and (11) in Wide Neural Networks of 
            Any Depth Evolve as Linear Models Under Gradient Descent (J. Lee and L. Xiao et al. 2019). 
            Paper link: https://arxiv.org/pdf/1902.06720.pdf.
    :param t: a scalar of array of scalars of any shape. `t=None` is treated as infinity and returns 
            the same result as `t=np.inf`, but is computed using identity or linear solve for train 
            and test predictions respectively instead of eigendecomposition, saving time and precision.
            Equivalent of training steps (but can be fractional).
    :param diag_reg: (optional) a scalar representing the strength of the diagonal regularization for `k_train_train`, 
            i.e. computing `k_train_train + diag_reg * I` during Cholesky factorization or eigendecomposition.
    :return: a np.ndarray for the model logits.
    """
    # Kernel
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')
    
    # Prediction
    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)
    return predict_fn(t, fx_train_0, fx_test_0, ntk_test_train)

def adv_loss(x_train, x_test, y_train, y_test, kernel_fn, loss='mse', t=None, targeted=False, diag_reg=1e-4):
    """
    :param x_train: input tensor (training data).
    :param x_test: input tensor (test data; used for evaluation).
    :param y_train: Tensor with one-hot true labels of training data.
    :param y_test: Tensor with one-hot true labels of test data. If targeted is true, then provide the
            target one-hot label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting poisoned data. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None. This argument does not have
            to be a binary one-hot label (e.g., [0, 1, 0, 0]), it can be floating points values
            that sum up to 1 (e.g., [0.05, 0.85, 0.05, 0.05]).
    :param kernel_fn: a callable that takes an input tensor and returns the kernel matrix.
    :param loss: loss function.
    :param t: a scalar of array of scalars of any shape. `t=None` is treated as infinity and returns 
            the same result as `t=np.inf`, but is computed using identity or linear solve for train 
            and test predictions respectively instead of eigendecomposition, saving time and precision.
            Equivalent of training steps (but can be fractional).
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
    :param diag_reg: (optional) a scalar representing the strength of the diagonal regularization for `k_train_train`, 
            i.e. computing `k_train_train + diag_reg * I` during Cholesky factorization or eigendecomposition.
    :return: a float for loss.
    """
    # Kernel
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')
    
    # Prediction
    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)
    fx = predict_fn(t, 0., 0., ntk_test_train)[1]
    
    # Loss
    if loss == 'cross-entropy':
        loss = cross_entropy_loss(fx, y_test)
    elif loss == 'mse':
        loss = mse_loss(fx, y_test)
        
    if targeted:
        loss = -loss        
    return loss