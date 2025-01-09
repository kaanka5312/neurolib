import jax
import jax.numpy as jnp


def variance_loss(output):
    """
    Args:
        output (jax.numpy.ndarray): Time series data with shape (n_output_vars, N, T)
            where N is number of nodes and T is number of timepoints

    Returns:
        float: Variance over time, averaged across output variables and nodes
    """
    return jnp.var(output, axis=(0, 1)).mean()


def cross_correlation_loss(output, dt=1.0):
    """
    Args:
        output (jax.numpy.ndarray): Time series data with shape (n_output_vars, N, T)
            where N is number of nodes and T is number of timepoints
        dt (float): Time step

    Returns:
        float: Negative cross-correlation
    """
    _, N, T = output.shape
    xmean = jnp.mean(output, axis=2, keepdims=True)
    xstd = jnp.std(output, axis=2, keepdims=True)

    xvec = (output - xmean) / xstd

    lossmat = jnp.einsum("vnt,vkt->vnkt", xvec, xvec)
    diag = jnp.einsum("vnt,vnt->vt", xvec, xvec)
    loss = jnp.sum(jnp.sum(lossmat, axis=(1, 2)) - diag) * dt / 2.0
    loss *= -2.0 / (N * (N - 1) * T * dt)
    return loss


def hilbert(signal, axis=-1):
    n = signal.shape[axis]
    h = jnp.zeros(n)
    h = h.at[0].set(1)

    if n % 2 == 0:
        h = h.at[1 : n // 2].set(2)
        h = h.at[n // 2].set(1)
    else:
        h = h.at[1 : (n + 1) // 2].set(2)

    h = jnp.expand_dims(h, tuple(i for i in range(signal.ndim) if i != axis))
    h = jnp.broadcast_to(h, signal.shape)

    fft_signal = jnp.fft.fft(signal, axis=axis)
    analytic_fft = fft_signal * h

    analytic_signal = jnp.fft.ifft(analytic_fft)
    return analytic_signal


def kuramoto_loss(output):
    """
    Args:
        output (jax.numpy.ndarray): Time series data with shape (n_output_vars, N, T)
            where N is number of nodes and T is number of timepoints

    Returns:
        float: Negative Kuramoto order parameter averaged over output variables
    """
    phase = jnp.angle(hilbert(output, axis=2))
    return -jnp.mean(jnp.real(jnp.mean(jnp.exp(1j * phase), axis=1)))


def get_fourier_component(data, target_frequency, dt=1.0):
    fourier_series = jnp.abs(jnp.fft.fft(data)[: len(data) // 2])
    freqs = jnp.fft.fftfreq(data.size, d=dt)[: len(data) // 2]
    return fourier_series[jnp.argmin(jnp.abs(freqs - target_frequency))]


def osc_fourier_loss(output, target_frequency, dt=1.0):
    """
    Args:
        output (jax.numpy.ndarray): Time series data with shape (n_output_vars, N, T)
            where N is number of nodes and T is number of timepoints
        target_frequency (float): Frequency to optimize for
        dt (float): Time step

    Returns:
        float: Negative synchronization of output nodes at target frequency, irrespective of phase
    """
    loss = 0.0
    for n in range(output.shape[1]):
        for v in range(output.shape[0]):
            loss -= get_fourier_component(output[v, n], target_frequency) ** 2
    return loss / (output.shape[2] * dt) ** 2


def sync_fourier_loss(output, target_frequency, dt=1.0):
    """
    Args:
        output (jax.numpy.ndarray): Time series data with shape (n_output_vars, N, T)
            where N is number of nodes and T is number of timepoints
        target_frequency (float): Frequency to optimize for
        dt (float): Time step

    Returns:
        float: Negative synchronization of output nodes at target frequency, considering phase
    """
    loss = 0.0
    for v in range(output.shape[0]):
        loss -= get_fourier_component(jnp.sum(output[v], axis=0), target_frequency) ** 2
    return loss / (output.shape[2] * dt) ** 2
