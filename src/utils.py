
import numpy as np

def geometric_brownian_motion(S0, mu, sigma, T, N, simulations=1, seed=None):
    """
    Simulate stock price paths using Geometric Brownian Motion (GBM).

    Parameters:
    - S0: Initial stock price
    - mu: Expected return (drift)
    - sigma: Volatility
    - T: Total time (in years)
    - N: Number of time steps
    - simulations: Number of paths to generate
    - seed: Random seed for reproducibility

    Returns:
    - A NumPy array of shape (simulations, N) with simulated price paths
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    prices = np.zeros((simulations, N))
    prices[:, 0] = S0
    
    for t in range(1, N):
        Z = np.random.normal(0, 1, simulations)
        prices[:, t] = prices[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * Z * np.sqrt(dt))
    
    return prices
