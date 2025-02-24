from functools import partial

import jax
import jax.numpy as jnp
from jax import random, lax, jit, grad, vmap, jacrev, hessian, pmap
from jax.tree_util import tree_map

import optax

from jaxpi import archs
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn

import copy


class GreyScott(ForwardIVP):
    def __init__(
        self, config, t_star, x_star, y_star, u0, v0, b1, b2, c1, c2, eps1, eps2
    ):
        super().__init__(config)

        self.u0 = u0
        self.v0 = v0

        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star

        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.eps1 = eps1
        self.eps2 = eps2

        # Predictions over a grid
        self.u0_pred_fn = vmap(
            vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)
        )
        self.v0_pred_fn = vmap(
            vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)
        )

        self.u_pred_fn = vmap(
            vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )
        self.v_pred_fn = vmap(
            vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)),
            (None, 0, None, None),
        )
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        t = t / self.t_star[-1]
        z = jnp.stack([t, x, y])
        _, outputs = self.state.apply_fn(params, z)

        u = outputs[0]
        v = outputs[1]
        return u, v

    def u_net(self, params, t, x, y):
        u, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v = self.neural_net(params, t, x, y)
        return v

    def r_net(self, params, t, x, y):
        u, v = self.neural_net(params, t, x, y)

        u_t, v_t = jacrev(self.neural_net, argnums=1)(params, t, x, y)

        u_hessian, v_hessian = hessian(self.neural_net, argnums=(2, 3))(params, t, x, y)

        u_xx = u_hessian[0][0]
        u_yy = u_hessian[1][1]

        v_xx = v_hessian[0][0]
        v_yy = v_hessian[1][1]

        ru = u_t - self.b1 * (1 - u) + self.c1 * u * v**2 - self.eps1 * (u_xx + u_yy)
        rv = v_t + self.b2 * v - self.c2 * u * v**2 - self.eps2 * (v_xx + v_yy)

        return ru, rv

    def ru_net(self, params, t, x, y):
        ru, _ = self.r_net(params, t, x, y)
        return ru

    def rv_net(self, params, t, x, y):
        _, rv = self.r_net(params, t, x, y)
        return rv

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        ru_pred, rv_pred = self.r_pred_fn(params, t_sorted, batch[:, 1], batch[:, 2])

        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rv_pred = rv_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rv_l = jnp.mean(rv_pred**2, axis=1)

        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rv_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rv_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rv_gamma])
        gamma = gamma.min(0)

        return ru_l, rv_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial conditions loss
        u0_pred = self.u0_pred_fn(params, 0.0, self.x_star, self.y_star)
        v0_pred = self.v0_pred_fn(params, 0.0, self.x_star, self.y_star)

        u0_loss = jnp.mean((u0_pred - self.u0) ** 2)
        v0_loss = jnp.mean((v0_pred - self.v0) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, gamma = self.res_and_w(params, batch)
            ru_loss = jnp.mean(ru_l * gamma)
            rv_loss = jnp.mean(rv_l * gamma)

        else:
            ru_pred, rv_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            # Compute loss
            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)

        loss_dict = {
            "u_ic": u0_loss,
            "v_ic": v0_loss,
            "ru": ru_loss,
            "rv": rv_loss,
        }
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        u_ic_ntk = vmap(
            vmap(ntk_fn, (None, None, None, None, 0)), (None, None, None, 0, None)
        )(self.u_net, params, 0.0, self.x_star, self.y_star)
        v_ic_ntk = vmap(
            vmap(ntk_fn, (None, None, None, None, 0)), (None, None, None, 0, None)
        )(self.v_net, params, 0.0, self.x_star, self.y_star)

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1], batch[:, 2]]).T
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.ru_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )

            ru_ntk = ru_ntk.reshape(self.num_chunks, -1)
            rv_ntk = rv_ntk.reshape(self.num_chunks, -1)

            ru_ntk = jnp.mean(ru_ntk, axis=1)
            rv_ntk = jnp.mean(rv_ntk, axis=1)

            _, _, casual_weights = self.res_and_w(params, batch)
            ru_ntk = ru_ntk * casual_weights
            rv_ntk = rv_ntk * casual_weights
        else:
            ru_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.ru_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            rv_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(
                self.rv_net, params, batch[:, 0], batch[:, 1], batch[:, 2]
            )

        ntk_dict = {"u_ic": u_ic_ntk, "v_ic": v_ic_ntk, "ru": ru_ntk, "rv": rv_ntk}
        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, x, y, u_ref, v_ref):
        u_pred = self.u_pred_fn(params, t, x, y)
        v_pred = self.v_pred_fn(params, t, x, y)

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)

        return u_error, v_error


class GreyScottEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, v_ref):
        u_error, v_error = self.model.compute_l2_error(
            params,
            self.model.t_star,
            self.model.x_star,
            self.model.y_star,
            u_ref,
            v_ref,
        )
        self.log_dict["u_error"] = u_error
        self.log_dict["v_error"] = v_error

    def __call__(self, state, batch, u_ref, v_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref, v_ref)

        if self.config.weighting.use_causal:
            _, _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_nonlinearities:
            layer_keys = [
                key
                for key in state.params["params"].keys()
                if key.endswith(
                    tuple(
                        [f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]
                    )
                )
            ]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params["params"][key]["alpha"]

        return self.log_dict
    
# ======================================================================================================== #

def pirate_get_dense_layers(params):
    layers = []
    for key1, param1 in params['params'].items():
        if 'PIModifiedBottleneck' in key1:
            for key2, param2 in param1.items():
                if 'Dense' in key2:
                    layers.append(param2)
    
    return layers


def make_mask(state):
    step = 0
    
    # need to adjust for model
    layers = pirate_get_dense_layers(state.params)
    
    for param in layers:
        step += 1

    mask = [None] * step
    
    step = 0
    for param in layers:
        value1 = param['kernel'][0]
        value2 = param['kernel'][1]

        mask[step] = (jnp.ones_like(value1), jnp.ones_like(value2))
        step += 1

    return mask


class LotteryTicketGreyScott(GreyScott):
    def __init__(
        self, config, t_star, x_star, y_star, u0, v0, b1, b2, c1, c2, eps1, eps2
    ):
        
        super().__init__(config, t_star, x_star, y_star, u0, v0, b1, b2, c1, c2, eps1, eps2)

        self.mask = make_mask(self.state)
        self.init_params = copy.deepcopy(self.state.params)

    
    def prune_by_percentile(self, state, mask, percent):
        step = 0

        layers = pirate_get_dense_layers(state.params)
        for param in layers:
            value1 = param['kernel'][0]
            value2 = param['kernel'][1]

            alive = jnp.concat([value1[jnp.nonzero(value1)], value2[jnp.nonzero(value2)]])

            percentile_value = jnp.percentile(abs(alive), percent)
            new_mask_1 = jnp.where(abs(value1) < percentile_value, 0, mask[step][0])
            new_mask_2 = jnp.where(abs(value2) < percentile_value, 0, mask[step][1])

            param['kernel'] = (value1 * new_mask_1, value2 * new_mask_2)
            mask[step] = (new_mask_1, new_mask_2)

            step += 1

        return state, mask
    

    def original_initialiaztion(self, state, mask):
        step = 0

        curr_layers = pirate_get_dense_layers(state.params)
        init_layers = pirate_get_dense_layers(self.init_params)

        for curr_param, init_param in zip(curr_layers, init_layers):
            if 'kernel' in curr_param:
                curr_param['kernel'] = (
                    mask[step][0] * init_param['kernel'][0],
                    mask[step][1] * init_param['kernel'][1]
                )
                step += 1

            if 'bias' in curr_param:
                curr_param['bias'] = init_param['bias']

        return state


    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch, *args):
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        grads = lax.pmean(grads, "batch")

        EPS = 1e-6
        
        param_layers = pirate_get_dense_layers(state.params)
        grad_layers = pirate_get_dense_layers(grads)

        for param, _grad in zip(param_layers, grad_layers):
            if 'kernel' in param:
                value1 = param['kernel'][0]
                value2 = param['kernel'][1]

                _grad['kernel'] = (
                    jnp.where(value1 < EPS, 0, _grad['kernel'][0]),
                    jnp.where(value2 < EPS, 0, _grad['kernel'][1])
                )
        
        state = state.apply_gradients(grads=grads)
        return state





