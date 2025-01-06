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


class OcWc:
    def __init__(self, model, target, optimizer=optax.adam(1e-3), opt_params=["exc_ext"]):
        self.model = copy.deepcopy(model)
        self.target = target
        self.opt_params = opt_params
        self.optimizer = optimizer
        
        self.weights = getdefaultweights()
        self.M = 1

        args_values = timeIntegration_args(self.model.params)
        self.args = dict(zip(args_names, args_values))

        self.loss = self.get_loss()
        self.compute_gradient = jax.jit(jax.grad(self.loss))
        self.T = len(self.args["t"]) + 1
        self.startind = self.model.getMaxDelay() + 1
        self.control = jnp.zeros((len(opt_params), self.model.params.N, self.T), dtype=float)  # TODO: depend on opt_params
        self.opt_state = self.optimizer.init(self.control)
        
        self.cost_history = []

        self.dim_vars = len(self.model.state_vars)
        self.dim_in = 1
        self.dim_out = len(self.model.output_vars)

    def simulate(self, control):
        args_local = self.args.copy()
        args_local.update(dict(zip(self.opt_params, [c for c in control])))
        return timeIntegration_elementwise(**args_local)

    def get_loss(self):
        @jax.jit
        def loss(control):
            t, exc, inh, exc_ou, inh_ou = self.simulate(control)
            return self.compute_total_cost(control, exc[:, self.startind - 1 :])

        return loss

    def accuracy_cost(self, exc):
        accuracy_cost = 0.0
        if self.weights["w_p"] != 0.0:
            accuracy_cost += self.weights["w_p"] * 0.5 * self.model.params.dt * jnp.sum((exc - self.target) ** 2)
        if self.weights["w_cc"] != 0.0:
            accuracy_cost += self.weights["w_cc"] * self.compute_cc_cost(exc)
        return accuracy_cost

    def control_strength_cost(self, control):
        control_strength_cost = 0.0
        if self.weights["w_2"] != 0.0:
            control_strength_cost += self.weights["w_2"] * 0.5 * self.model.params.dt * jnp.sum(control**2)
        if self.weights["w_1D"] != 0.0:
            control_strength_cost += self.compute_ds_cost(control)
        return control_strength_cost

    def compute_ds_cost(self, control):
        return jnp.sum(jnp.sqrt(jnp.sum(control**2, axis=1) * self.model.params.dt), axis=0)

    def compute_cc_cost(self, exc):
        xmean = jnp.stack([jnp.mean(exc, axis=1)] * exc.shape[1]).T
        xstd = jnp.stack([jnp.std(exc, axis=1)] * exc.shape[1]).T
        N = self.model.params.N

        xvec = (exc - xmean) / xstd

        costmat = jnp.einsum("ik,jk->ijk", xvec, xvec)
        diag = jnp.einsum("ij,ij->j", xvec, xvec)
        cost = np.sum(np.sum(np.sum(costmat, axis=0), axis=0) - diag, axis=0) * self.model.params.dt / 2.0
        cost *= -2.0 / (N * (N - 1) * (self.T) * self.model.params.dt)

        return cost

    def compute_total_cost(self, control, exc):
        """Compute the total cost as weighted sum precision of all contributing cost terms.
        :rtype: float
        """
        accuracy_cost = self.accuracy_cost(jnp.array(exc))
        control_strength_cost = self.control_strength_cost(control)
        return accuracy_cost + control_strength_cost

    def optimize_deterministic(self, n_max_iterations, output_every_nth=None):
        """Compute the optimal control signal for noise averaging method 0 (deterministic, M=1).

        :param n_max_iterations: maximum number of iterations of gradient descent
        :type n_max_iterations: int
        """

        # (I) forward simulation
        t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)  # yields x(t)

        cost = self.compute_total_cost(self.control, exc[:, self.startind - 1 :])
        print(f"Cost in iteration 0: %s" % (cost))
        if len(self.cost_history) == 0:  # add only if control model has not yet been optimized
            self.cost_history.append(cost)

        for i in range(1, n_max_iterations + 1):
            self.gradient = self.compute_gradient(self.control)

            updates, self.opt_state = self.optimizer.update(self.gradient, self.opt_state)
            self.control = optax.apply_updates(self.control, updates)

            t, exc, inh, exc_ou, inh_ou = self.simulate(self.control)

            cost = self.compute_total_cost(self.control, exc[:, self.startind - 1 :])
            if output_every_nth is not None and i % output_every_nth == 0:
                print(f"Cost in iteration %s: %s" % (i, cost))
            self.cost_history.append(cost)

        print(f"Final cost : %s" % (cost))

