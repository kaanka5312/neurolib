import jax
import jax.numpy as jnp
import numpy as np
import optax
import copy
from neurolib.models.jax.wc import WCModel
from neurolib.models.jax.wc.timeIntegration import timeIntegration_args, timeIntegration_elementwise

import logging
from neurolib.control.optimal_control.oc import getdefaultweights

# TODO: introduce for all models, not just WC
wc_default_control_params = ["exc_ext", "inh_ext"]
wc_default_target_params = ["exc", "inh"]


class Optimize:
    def __init__(
        self,
        model,
        loss_function,
        param_names,
        target_param_names,
        target=None,
        init_params=None,
        optimizer=optax.adabelief(1e-3),
    ):
        assert isinstance(param_names, (list, tuple)) and len(param_names) > 0
        assert isinstance(target_param_names, (list, tuple)) and len(target_param_names) > 0
        assert all([p in model.args_names for p in param_names])
        assert all([tp in model.output_vars for tp in target_param_names])

        self.model = copy.deepcopy(model)
        self.loss_function = loss_function
        self.target = target
        self.optimizer = optimizer
        self.param_names = param_names
        self.target_param_names = target_param_names

        args_values = timeIntegration_args(self.model.params)
        self.args = dict(zip(self.model.args_names, args_values))

        self.T = len(self.args["t"])
        self.startind = self.model.getMaxDelay()
        if init_params is not None:
            self.params = init_params
        else:
            self.params = dict(zip(param_names, [self.args[p] for p in param_names]))
        self.opt_state = self.optimizer.init(self.params)

        compute_loss = lambda params: self.loss_function(params, self.get_output(params))
        self.compute_loss = jax.jit(compute_loss)
        self.compute_gradient = jax.jit(jax.grad(self.compute_loss))

        self.cost_history = []

    # TODO: allow arbitrary model, not just WC
    def simulate(self, params):
        args_local = self.args.copy()
        args_local.update(params)
        t, exc, inh, exc_ou, inh_ou = timeIntegration_elementwise(**args_local)
        return {
            "t": t,
            "exc": exc,
            "inh": inh,
            "exc_ou": exc_ou,
            "inh_ou": inh_ou,
        }

    def get_output(self, params):
        simulation_results = self.simulate(params)
        return jnp.stack([simulation_results[tp][:, self.startind :] for tp in self.target_param_names])

    def get_loss(self):
        @jax.jit
        def loss(params):
            output = self.get_output(params)
            return self.loss_function(params, output)

        return loss

    def optimize_deterministic(self, n_max_iterations, output_every_nth=None):
        loss = self.compute_loss(self.control)
        print(f"loss in iteration 0: %s" % (loss))
        if len(self.cost_history) == 0:  # add only if control model has not yet been optimized
            self.cost_history.append(loss)

        for i in range(1, n_max_iterations + 1):
            self.gradient = self.compute_gradient(self.control)
            updates, self.opt_state = self.optimizer.update(self.gradient, self.opt_state)
            self.control = optax.apply_updates(self.control, updates)

            if output_every_nth is not None and i % output_every_nth == 0:
                loss = self.compute_loss(self.control)
                self.cost_history.append(loss)
                print(f"loss in iteration %s: %s" % (i, loss))

        loss = self.compute_loss(self.control)
        print(f"Final loss : %s" % (loss))


class OcWc(Optimize):
    def __init__(
        self,
        model,
        target=None,
        optimizer=optax.adabelief(1e-3),
        control_param_names=wc_default_control_params,
        target_param_names=wc_default_target_params,
    ):
        super().__init__(
            model,
            self.compute_total_cost,
            control_param_names,
            target_param_names,
            target=target,
            init_params=None,
            optimizer=optimizer,
        )
        self.control = self.params
        self.weights = getdefaultweights()

    def compute_total_cost(self, control, output):
        """
        Compute the total cost as the sum of accuracy cost and control strength cost.

        Parameters:
        control (dict[str, jax.numpy.ndarray]): Dictionary of control inputs, where each entry has shape (N, T).
        output (jax.numpy.ndarray): Simulation output of shape ((len(target_param_names)), N, T).

        Returns:
        float: The total cost.
        """
        accuracy_cost = self.accuracy_cost(output)
        control_arr = jnp.array(list(control.values()))
        control_strength_cost = self.control_strength_cost(control_arr)
        return accuracy_cost + control_strength_cost

    # TODO: move cost functions outside
    def accuracy_cost(self, output):
        accuracy_cost = 0.0
        if self.weights["w_p"] != 0.0:
            accuracy_cost += self.weights["w_p"] * 0.5 * self.model.params.dt * jnp.sum((output - self.target) ** 2)
        if self.weights["w_cc"] != 0.0:
            accuracy_cost += self.weights["w_cc"] * self.compute_cc_cost(output)
        if self.weights["w_var"] != 0.0:
            accuracy_cost += self.weights["w_var"] * self.compute_var_cost(output)
        if self.weights["w_f_osc"] != 0.0:
            accuracy_cost += self.weights["w_f_osc"] * self.compute_osc_fourier_cost(output)
        return accuracy_cost

    def control_strength_cost(self, control):
        control_strength_cost = 0.0
        if self.weights["w_2"] != 0.0:
            control_strength_cost += self.weights["w_2"] * 0.5 * self.model.params.dt * jnp.sum(control**2)
        if self.weights["w_1D"] != 0.0:
            control_strength_cost += self.weights["w_1D"] * self.compute_ds_cost(control)
        return control_strength_cost

    def compute_ds_cost(self, control):
        eps = 1e-6  # avoid grad(sqrt(0.0))
        return jnp.sum(jnp.sqrt(jnp.sum(control**2, axis=2) * self.model.params.dt + eps))

    def compute_cc_cost(self, output):
        xmean = jnp.mean(output, axis=2, keepdims=True)
        xstd = jnp.std(output, axis=2, keepdims=True)

        xvec = (output - xmean) / xstd

        costmat = jnp.einsum("vnt,vkt->vnkt", xvec, xvec)
        diag = jnp.einsum("vnt,vnt->vt", xvec, xvec)
        cost = jnp.sum(jnp.sum(costmat, axis=(1, 2)) - diag) * self.model.params.dt / 2.0
        cost *= -2.0 / (self.model.params.N * (self.model.params.N - 1) * self.T * self.model.params.dt)
        return cost

    def compute_var_cost(self, output):
        return jnp.var(output, axis=(0, 1)).mean()

    def compute_osc_fourier_cost(self, output): ...
