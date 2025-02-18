import os
import time

import numpy as np
import scipy

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

import ml_collections
import wandb

from jaxpi.archs import Embedding
from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, save_config

import models
from utils import get_dataset


def train_one_window(config, workdir, model, res_sampler, u_ref, idx):
    logger = Logger()

    evaluator = models.KSEvaluator(config, model)

    step_offset = idx * config.training.max_steps

    print("Waiting for JIT...")
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step + step_offset)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Save model checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    path = os.path.join(
                    workdir, config.wandb.name, "config"
                )
    save_config(config, path)

    # Get the reference solution
    u_ref, t_star, x_star = get_dataset(config.time_fraction)

    k = 0
    u0 = u_ref[k, :]  # initial condition of the first time window
    u_ref = u_ref[k:, :]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Define the time and space domain
    dt = t[1] - t[0]
    t0 = t[0]
    t1 = (
        t[-1] + 2 * dt
    )  # cover the start point of the next time window, which is t_star[num_time_steps]

    x0 = x_star[0]
    x1 = x_star[-1]
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Initialize the residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size))

    for idx in range(config.training.num_time_windows):
        print("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :]

        if config.use_pi_init:
            print("Use physics-informed initialization...")

            model = models.KS(config, u0, t_star, x_star)
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params

            t_scaled = t[::5] / t[-1]

            tt, xx = jnp.meshgrid(t_scaled, x_star, indexing='ij')
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
            u_linear = jnp.tile(u0.flatten(), (t_scaled.shape[0], 1))

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

            coeffs, residuals, rank, s = jnp.linalg.lstsq(feat_matrix, u_linear.flatten(), rcond=None)
            print("least square residuals: ", residuals)

            config.arch.pi_init = coeffs.reshape(-1, 1)

            del model, state, params

        # Initialize the model
        model = models.KS(config, u0, t, x_star)

        # Training the current time window
        model = train_one_window(config, workdir, model, res_sampler, u, idx)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            u0 = vmap(model.u_net, (None, None, 0))(
                params, t_star[num_time_steps], x_star
            )

            del model, state, params
