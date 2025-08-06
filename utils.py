import numpy as np
from scipy.stats import norm


def mag_likelihood(m_obs, mu, ms, sigma_m):
    """Likelihood of an observed magnitude given magnification and source magnitude."""
    mag_model = ms - 2.5 * np.log10(np.abs(mu))
    return norm.pdf(m_obs, loc=mag_model, scale=sigma_m)

__all__ = ["mag_likelihood"]
