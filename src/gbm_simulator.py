
import matplotlib.pyplot as plt
from utils import geometric_brownian_motion

def run_simulation():
    # Parameters
    S0 = 100          # Initial stock price
    mu = 0.05         # Drift
    sigma = 0.2       # Volatility
    T = 1.0           # Time horizon (in years)
    N = 252           # Number of time steps
    simulations = 1000
    seed = 42

    # Generate simulations
    predicted_paths = geometric_brownian_motion(S0, mu, sigma, T, N, simulations, seed)
    mean_path = predicted_paths.mean(axis=0)

    # Plot
    plt.figure(figsize=(12, 6))
    for i in range(min(simulations, 100)):
        plt.plot(predicted_paths[i], color='gray', alpha=0.1)
    plt.plot(mean_path, label='Mean Prediction', color='blue', linewidth=2)
    plt.title("Monte Carlo Simulated Stock Price Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
