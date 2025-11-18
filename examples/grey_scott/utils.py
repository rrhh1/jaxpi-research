import scipy.io
import jax
import jax.numpy as jnp

def get_dataset(fraction):
    data = scipy.io.loadmat("data/grey_scott_2.mat")

    u_ref = data["usol"]
    v_ref = data["vsol"]

    t_star = data["t"].flatten()
    x_star = data["x"].flatten()
    y_star = data["y"].flatten()

    start_time_step = int(fraction[0] * len(t_star))
    end_time_step = int(fraction[1] * len(t_star))

    u_ref = u_ref[start_time_step:end_time_step, :, :]
    v_ref = v_ref[start_time_step:end_time_step, :, :]
    t_star = t_star[: end_time_step - start_time_step]

    b1 = data["b1"].flatten()[0]
    b2 = data["b2"].flatten()[0]

    c1 = data["c1"].flatten()[0]
    c2 = data["c2"].flatten()[0]

    eps1 = data["ep1"].flatten()[0]
    eps2 = data["ep2"].flatten()[0]

    return u_ref, v_ref, t_star, x_star, y_star, b1, b2, c1, c2, eps1, eps2


def print_ratios(params, idx, step):
    dense_layers = {
        "Dense_0": params["params"]["Dense_0"],
        "Dense_1": params["params"]["Dense_1"],

        "PIModifiedBottleneck_0_Dense_0": params["params"]["PIModifiedBottleneck_0"]["Dense_0"],
        "PIModifiedBottleneck_0_Dense_1": params["params"]["PIModifiedBottleneck_0"]["Dense_1"],
        "PIModifiedBottleneck_0_Dense_2": params["params"]["PIModifiedBottleneck_0"]["Dense_2"],

        "PIModifiedBottleneck_1_Dense_0": params["params"]["PIModifiedBottleneck_1"]["Dense_0"],
        "PIModifiedBottleneck_1_Dense_1": params["params"]["PIModifiedBottleneck_1"]["Dense_1"],
        "PIModifiedBottleneck_1_Dense_2": params["params"]["PIModifiedBottleneck_1"]["Dense_2"],

        "PIModifiedBottleneck_2_Dense_0": params["params"]["PIModifiedBottleneck_2"]["Dense_0"],
        "PIModifiedBottleneck_2_Dense_1": params["params"]["PIModifiedBottleneck_2"]["Dense_1"],
        "PIModifiedBottleneck_2_Dense_2": params["params"]["PIModifiedBottleneck_2"]["Dense_2"],
    }

    def binary_step(input):
        return jnp.float32(input > 0)


    file = open(f"ratio_logs/ratios_window_{idx}.txt", "a")
    file.write(f"Step {step}\n")

    total = 0.
    keep = 0.

    for key, layer in dense_layers.items():
        abs_kernel = jnp.abs(layer["kernel"][1])
        threshold_value = jnp.reshape(layer["threshold"], (abs_kernel.shape[0], 1))
        abs_kernel = abs_kernel - threshold_value

        mask = binary_step(abs_kernel)
        ratio = jnp.sum(mask) / mask.size

        def create_new_mask(threshold_value):
            abs_kernel = jnp.abs(layer["kernel"][1])
            new_threshold_value = jnp.reshape(threshold_value, (abs_kernel.shape[0], 1))
            abs_kernel = abs_kernel - new_threshold_value

            return binary_step(abs_kernel)

        threshold = jax.lax.cond(
            ratio <= 0.01,
            lambda x: jnp.zeros_like(x),
            lambda x: x,
            layer["threshold"]
        )

        mask = jnp.where(ratio <= 0.01, create_new_mask(threshold), mask)
        ratio = jnp.sum(mask) / mask.size

        total += mask.size
        keep += jnp.sum(mask)

        file.write(key + " ratio: " + str(ratio) + "\n")

    pi_init = params["params"]["pi_init"]
    threshold = params["params"]["threshold"]

    abs_kernel = jnp.abs(pi_init)
    threshold_value = jnp.reshape(threshold, (abs_kernel.shape[0], 1))
    abs_kernel = abs_kernel - threshold_value

    mask = binary_step(abs_kernel)
    ratio = jnp.sum(mask) / mask.size

    def create_new_mask(threshold_value):
        abs_kernel = jnp.abs(pi_init)
        new_threshold_value = jnp.reshape(threshold_value, (abs_kernel.shape[0], 1))
        abs_kernel = abs_kernel - new_threshold_value

        return binary_step(abs_kernel)

    threshold = jax.lax.cond(
        ratio <= 0.01,
        lambda x: jnp.zeros_like(x),
        lambda x: x,
        threshold
    )

    mask = jnp.where(ratio <= 0.01, create_new_mask(threshold), mask)
    ratio = jnp.sum(mask) / mask.size

    total += mask.size
    keep += jnp.sum(mask)

    file.write("Pi_init ratio: " + str(ratio) + "\n")

    file.write("Overall ratio: " + str(keep / total) + "\n")
    file.write("=" * 30 + "\n")
    file.close()