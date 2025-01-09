import jax
import jax.numpy as jnp
import numpy as np
import optax
import copy
from neurolib.models.jax.wc import WCModel
from neurolib.models.jax.wc.timeIntegration import timeIntegration_args, timeIntegration_elementwise
from neurolib.optimize.loss_functions import (
    kuramoto_loss,
    cross_correlation_loss,
    variance_loss,
    osc_fourier_loss,
    sync_fourier_loss,
)
from neurolib.control.optimal_control.oc import getdefaultweights


class Optimize:
    def __init__(
        self,
        model,
        loss_function,
        param_names,
        target_param_names,
        init_params=None,
        regularization_function=lambda _: 0.0,
        optimizer=optax.adabelief(1e-3),
    ):
        assert isinstance(param_names, (list, tuple)) and len(param_names) > 0
        assert isinstance(target_param_names, (list, tuple)) and len(target_param_names) > 0
        assert all([p in model.args_names for p in param_names])
        assert all([tp in model.output_vars for tp in target_param_names])

        self.model = copy.deepcopy(model)
        self.loss_function = loss_function
        self.regularization_function = regularization_function
        self.optimizer = optimizer
        self.param_names = param_names
        self.target_param_names = target_param_names

        args_values = timeIntegration_args(self.model.params)
        self.args = dict(zip(self.model.args_names, args_values))

        self.T = len(self.args["t"])
        self.startind = self.model.getMaxDelay() + 1
        if init_params is not None:
            self.params = init_params
        else:
            self.params = dict(zip(param_names, [self.args[p] for p in param_names]))
        self.opt_state = self.optimizer.init(self.params)

        # TODO: instead apply individually to each param
        compute_loss = lambda params: self.loss_function(
            jnp.stack(list(self.get_output(params).values()))
        ) + self.regularization_function(params)
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
        return {tp: simulation_results[tp][:, self.startind :] for tp in self.target_param_names}

    def optimize(self, n_max_iterations, output_every_nth=None):
        loss = self.compute_loss(self.params)
        print(f"loss in iteration 0: %s" % (loss))
        if len(self.cost_history) == 0:  # add only if params have not yet been optimized
            self.cost_history.append(loss)

        for i in range(1, n_max_iterations + 1):
            self.gradient = self.compute_gradient(self.params)
            updates, self.opt_state = self.optimizer.update(self.gradient, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

            if output_every_nth is not None and i % output_every_nth == 0:
                loss = self.compute_loss(self.params)
                self.cost_history.append(loss)
                print(f"loss in iteration %s: %s" % (i, loss))

        loss = self.compute_loss(self.params)
        print(f"Final loss : %s" % (loss))


class Oc(Optimize):
    """
    Convenience class for optimal control. The cost functional is constructed as a weighted sum of accuracy and control strength costs. Requires optimization parameters to be of shape (N, T).
    """

    supported_cost_parameters = [
        "w_p",
        "w_cc",
        "w_var",
        "w_f_osc",
        "w_f_sync",
        "w_ko",
        "w_2",
        "w_1D",
    ]

    def __init__(
        self,
        model,
        target_timeseries=None,
        target_frequency=None,
        optimizer=optax.adabelief(1e-3),
        control_param_names=["exc_ext", "inh_ext"],
        target_param_names=["exc", "inh"],
        weights=None,
    ):
        super().__init__(
            model,
            self.accuracy_cost,
            control_param_names,
            target_param_names,
            init_params=None,
            optimizer=optimizer,
            regularization_function=self.control_strength_cost,
        )
        self.target_timeseries = target_timeseries
        self.target_frequency = target_frequency
        self.control = self.params
        if weights is None:
            self.weights = getdefaultweights()

    def accuracy_cost(self, output):
        """
        Args:
            output (jax.numpy.ndarray): Simulation output of shape ((len(target_param_names)), N, T).
        """
        accuracy_cost = 0.0
        if self.weights["w_p"] != 0.0:
            accuracy_cost += self.weights["w_p"] * self.precision_cost(output)
        if self.weights["w_cc"] != 0.0:
            accuracy_cost += self.weights["w_cc"] * cross_correlation_loss(output, self.model.params.dt)
        if self.weights["w_var"] != 0.0:
            accuracy_cost += self.weights["w_var"] * variance_loss(output)
        if self.weights["w_f_osc"] != 0.0:
            accuracy_cost += self.weights["w_f_osc"] * osc_fourier_loss(
                output, self.target_frequency, self.model.params.dt
            )
        if self.weights["w_f_sync"] != 0.0:
            accuracy_cost += self.weights["w_f_sync"] * sync_fourier_loss(
                output, self.target_frequency, self.model.params.dt
            )
        if self.weights["w_ko"] != 0.0:
            accuracy_cost += self.weights["w_ko"] * kuramoto_loss(output)
        return accuracy_cost

    def precision_cost(self, output):
        return 0.5 * self.model.params.dt * jnp.sum((output - self.target_timeseries) ** 2)

    def control_strength_cost(self, control):
        """
        Args:
            control (dict[str, jax.numpy.ndarray]): Dictionary of control inputs, where each entry has shape (N, T).
        """
        control_arr = jnp.array(list(control.values()))
        control_strength_cost = 0.0
        if self.weights["w_2"] != 0.0:
            control_strength_cost += self.weights["w_2"] * 0.5 * self.model.params.dt * jnp.sum(control_arr**2)
        if self.weights["w_1D"] != 0.0:
            control_strength_cost += self.weights["w_1D"] * self.compute_ds_cost(control_arr)
        return control_strength_cost

    def compute_ds_cost(self, control):
        eps = 1e-6  # avoid grad(sqrt(0.0))
        return jnp.sum(jnp.sqrt(jnp.sum(control**2, axis=2) * self.model.params.dt + eps))

    def optimize(self, *args, **kwargs):
        super().optimize(*args, **kwargs)
        self.control = self.params
