import numpy as np
from scipy.stats import norm

def run_proportion_test(p_hat, q_hat, n_s, n_t, alpha=0.05):
    """
    Runs a two-sample proportion test comparing p_hat vs q_hat.
    Returns the z-statistic, p-value, and prints the decision.

    :param p_hat: Proportion from sample 1 (e.g., training).
    :param q_hat: Proportion from sample 2 (e.g., test).
    :param n_s: Size of sample 1.
    :param n_t: Size of sample 2.
    :param alpha: Significance level for hypothesis testing.
    :return: (z_stat, p_value)
    """
    # Pooled proportion
    a = (n_s * p_hat + n_t * q_hat) / (n_s + n_t)

    # Z-statistic
    z_stat = (q_hat - p_hat) / np.sqrt(a * (1 - a) * (1 / n_s + 1 / n_t))

    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return z_stat, p_value