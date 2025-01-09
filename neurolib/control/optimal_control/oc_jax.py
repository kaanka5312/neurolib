import jax
import jax.numpy as jnp
import numpy as np
import optax
import copy
from neurolib.models.jax.wc import WCModel
from neurolib.models.jax.wc.timeIntegration import timeIntegration_args, timeIntegration_elementwise
from neurolib.optimize.autodiff.wc_optimizer import args_names

import logging
from neurolib.control.optimal_control.oc import getdefaultweights

wc_default_control_params = ["exc_ext", "inh_ext"]
wc_default_target_params = ["exc", "inh"]

def hilbert_jax(signal, axis=-1): 

    n = signal.shape[axis]
    h = jnp.zeros(n)
    h = h.at[0].set(1)

    if n % 2 == 0:
        h = h.at[1:n//2].set(2)     
        h = h.at[n//2].set(1)      
    else:
        h = h.at[1:(n+1)//2].set(2)

    h = jnp.expand_dims(h, tuple(i for i in range(signal.ndim) if i != axis))
    h = jnp.broadcast_to(h, signal.shape)
    
    fft_signal = jnp.fft.fft(signal, axis=axis)
    analytic_fft = fft_signal * h
    
    analytic_signal = jnp.fft.ifft(analytic_fft)
    return analytic_signal


class OcWc:
    def __init__(
        self,
        model,
        target=None,
        optimizer=optax.adam(1e-3),
        control_params=wc_default_control_params,
        target_params=wc_default_target_params,
    ):
        assert isinstance(control_params, (list, tuple)) and len(control_params) > 0
        assert isinstance(target_params, (list, tuple)) and len(target_params) > 0
        assert all([cp in wc_default_control_params for cp in control_params])
        assert all([tp in wc_default_target_params for tp in target_params])

        self.model = copy.deepcopy(model)
        self.target = target
        self.optimizer = optimizer
        self.control_params = control_params
        self.target_params = target_params

        self.weights = getdefaultweights()

        args_values = timeIntegration_args(self.model.params)
        self.args = dict(zip(args_names, args_values))

        self.loss = self.get_loss()
        self.compute_gradient = jax.jit(jax.grad(self.loss))
        self.T = len(self.args["t"])
        self.startind = self.model.getMaxDelay()
        self.control = jnp.zeros((len(control_params), self.model.params.N, self.T), dtype=float)
        self.opt_state = self.optimizer.init(self.control)

        self.cost_history = []

    def simulate(self, control):
        args_local = self.args.copy()
        args_local.update(dict(zip(self.control_params, [c for c in control])))
        return timeIntegration_elementwise(**args_local)

    def get_output(self, control):
        t, exc, inh, exc_ou, inh_ou = self.simulate(control)
        if self.target_params == ["exc", "inh"]:
            output = jnp.stack((exc, inh), axis=0)
        elif self.target_params == ["exc"]:
            output = exc[None, ...]
        elif self.target_params == ["inh"]:
            output = inh[None, ...]
        return output[:, :, self.startind :]

    def get_loss(self):
        @jax.jit
        def loss(control):
            output = self.get_output(control)
            return self.compute_total_cost(control, output)

        return loss

    def compute_total_cost(self, control, output):
        """
        Compute the total cost as the sum of accuracy cost and control strength cost.

        Parameters:
        control (jax.numpy.ndarray): Control input array of shape ((len(control_params)), N, T).
        output (jax.numpy.ndarray): Simulation output of shape ((len(target_params)), N, T).

        Returns:
        float: The total cost.
        """
        accuracy_cost = self.accuracy_cost(output)
        control_strength_cost = self.control_strength_cost(control)
        return accuracy_cost + control_strength_cost

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
        if self.weights["w_f_sync"] != 0.0:
            accuracy_cost += self.weights["w_f_sync"] * self.compute_sync_fourier_cost(output)
        if self.weights["w_kuramoto"] != 0.0:
            accuracy_cost += self.weights["w_kuramoto"] * self.compute_kuramoto_cost(output)
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
    
    def get_fourier_component(self, data, target_period):
        fourier_series = jnp.abs(jnp.fft.fft(data)[:len(data)//2])
        freqs = jnp.fft.fftfreq(data.size,d=self.model.params.dt)[:len(data)//2]
        return fourier_series[jnp.argmin(jnp.abs(freqs - 1./target_period))]

    def compute_osc_fourier_cost(self, output):
        cost = 0.0
        for n in range(output.shape[1]):
            for v in range(output.shape[0]):
                cost -= self.get_fourier_component(output[v, n], self.target)**2
        return cost / (output.shape[2] * self.model.params.dt)**2
    
    def compute_sync_fourier_cost(self, output):
        cost = 0.0
        for v in range(output.shape[0]):
            cost -= self.get_fourier_component(jnp.sum(output[v], axis=0), self.target)**2
        return cost / (output.shape[2] * self.model.params.dt)**2

    def compute_kuramoto_cost(self, output):
        phase = jnp.angle(hilbert_jax(output, axis=2))
        return -jnp.mean(jnp.abs(jnp.mean(jnp.exp(complex(0,1)*phase), axis=1)))
    

    def optimize_deterministic(self, n_max_iterations, output_every_nth=None):
        """Compute the optimal control signal for noise averaging method 0.

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """

        output = self.get_output(self.control)

        cost = self.compute_total_cost(self.control, output)
        print(f"Cost in iteration 0: %s" % (cost))
        if len(self.cost_history) == 0:  # add only if control model has not yet been optimized
            self.cost_history.append(cost)

        for i in range(1, n_max_iterations + 1):
            self.gradient = self.compute_gradient(self.control)

            updates, self.opt_state = self.optimizer.update(self.gradient, self.opt_state)
            self.control = optax.apply_updates(self.control, updates)

            output = self.get_output(self.control)
            if output_every_nth is not None and i % output_every_nth == 0:
                cost = self.compute_total_cost(self.control, output)
                self.cost_history.append(cost)
                print(f"Cost in iteration %s: %s" % (i, cost))

        print(f"Final cost : %s" % (cost))
