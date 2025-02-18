import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Regression"
    wandb.name = "default"
    wandb.tag = None

    config.deriv_order = 0

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = None
    arch.reparam = None
    arch.pi_init = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 1000
    optim.staircase = False
    optim.warmup_steps = 0
    optim.grad_accum_steps = 0

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None
    weighting.init_weights = ml_collections.ConfigDict({"mse": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # # Input shape for initializing Flax models
    config.input_dim = 1

    # Integer for PRNG random seed.
    config.seed = 42

    return config
