import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Grey_Scott"
    wandb.name = "lottery_ticket_pirate"
    wandb.tag = None

    # Set the fractional size of the full temporal domain
    config.time_fraction = [0.0, 1.0]

    config.use_pi_init = True

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 3
    arch.hidden_dim = 256
    arch.out_dim = 2
    arch.activation = "swish"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (jnp.pi, jnp.pi), "axis": (1, 2), "trainable": (False, False)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )
    arch.nonlinearity = 0.0
    arch.pi_init = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.staircase = False
    optim.warmup_steps = 5000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size_per_device = 4096
    training.num_time_windows = 10

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"u_ic": 1.0, "v_ic": 1.0, "ru": 1.0, "rv": 1.0}
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_nonlinearities = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    # Lottery Ticket Hyperparameters
    config.lt = lt = ml_collections.ConfigDict()
    lt.prune_percentage = 20
    lt.prune_every_step = 1000
    

    return config
