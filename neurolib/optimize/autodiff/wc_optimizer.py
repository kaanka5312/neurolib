from jax import jit
from neurolib.models.jax.wc.timeIntegration import timeIntegration_args, timeIntegration_elementwise


# example usage:
# model = WCModel()
# wc_loss = get_loss(model.params, loss_f, ['exc_ext'])
# grad_wc_loss = jax.jit(jax.grad(wc_loss))
# grad_wc_loss([exc_ext])
def get_loss(model_params, loss_f, opt_params):
    args_values = timeIntegration_args(model_params)
    args = dict(zip(args_names, args_values))

    @jit
    def loss(x):
        args_local = args.copy()
        args_local.update(dict(zip(opt_params, x)))
        simulation_outputs = timeIntegration_elementwise(**args_local)
        return loss_f(x, *simulation_outputs)

    return loss
