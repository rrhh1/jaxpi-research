import os

from absl import logging
import ml_collections

import jax.numpy as jnp

from jaxpi.utils import restore_checkpoint
import models

from utils import get_dataset
import copy

def weight_test(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, t_star, x_star, y_star, b1, b2, c1, c2, eps1, eps2 = get_dataset(
        config.time_fraction
    )
    
    # Remove the last time step
    u_ref = u_ref[:-1, :]
    v_ref = v_ref[:-1, :]

    u0 = u_ref[0, :, :]
    v0 = v_ref[0, :, :]

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.GreyScott(config, t, x_star, y_star, u0, v0, b1, b2, c1, c2, eps1, eps2)

    idx = 9
    # Get the reference solution for the current time window
    u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
    v_star = v_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]

    # Restore the checkpoint
    ckpt_path = os.path.join(
        os.getcwd(), config.wandb.name, "ckpt", "time_window_{}".format(idx + 1)
    )
    
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params
    init_params = copy.deepcopy(params)

    weight_test_dict = {}
    layers = [
        'Dense_0',
        'Dense_1',

        'PIModifiedBottleneck_0,Dense_0',
        'PIModifiedBottleneck_0,Dense_1',
        'PIModifiedBottleneck_0,Dense_2',

        'PIModifiedBottleneck_1,Dense_0',
        'PIModifiedBottleneck_1,Dense_1',
        'PIModifiedBottleneck_1,Dense_2',

        'PIModifiedBottleneck_2,Dense_0',
        'PIModifiedBottleneck_2,Dense_1',
        'PIModifiedBottleneck_2,Dense_2',
    ]

    for layer in layers:
        layer_list = layer.split(',')

        param = params['params']
        for key in layer_list:
            param = param[key]

    
        value1 = param['kernel'][0]
        value2 = param['kernel'][1]

        for i in range(256):
            value1_modified = value1.at[i].set(0)
            param['kernel'] = (value1_modified, value2)

            u_error, v_error = model.compute_l2_error(
                params, t, x_star, y_star, u_star, v_star
            )

            key_name = layer + ",0," + str(i)
            weight_test_dict[key_name] = (u_error, v_error)

        for i in range(256):
            for j in range(256):
                value2_modified = value2.at[i, j].set(0)
                param['kernel'] = (value1, value2_modified)

                u_error, v_error = model.compute_l2_error(
                    params, t, x_star, y_star, u_star, v_star
                )

                key_name = layer + ",1," + str(i) + "-" + str(j)
                weight_test_dict[key_name] = (u_error, v_error)

        param['kernel'] = (value1, value2)

    weight_test_keys = list(weight_test_dict.keys())
    weight_test_keys.sort(key= lambda k : weight_test_dict[k][0] + weight_test_dict[k][1])

    for key in weight_test_keys:
        print(f"{key}: {weight_test_dict[key]}")