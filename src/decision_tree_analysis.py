
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def analyze_feature_importance(paths, mu, sigma):
    """
    Fit a Decision Tree Regressor to predict final prices based on simulation parameters.

    Parameters:
    - paths: Simulated price paths (array of shape: simulations x time steps)
    - mu: Drift value used in simulation
    - sigma: Volatility value used in simulation
    """

    final_prices = paths[:, -1]
    num_paths = paths.shape[0]

    # Example feature set: [initial price, mu, sigma, random_noise_std]
    X = np.array([[mu, sigma] for _ in range(num_paths)])
    y = final_prices

    # Create and train the Decision Tree model
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)

    # Feature importance
    feature_names = ['Drift (mu)', 'Volatility (sigma)']
    importances = model.feature_importances_

    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, importances, color='skyblue')
    plt.title("Feature Importance from Decision Tree Regressor")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

if __name__ == "__main__":
    # Dummy example using GBM-like final prices
    from utils import geometric_brownian_motion

    S0 = 100
    mu = 0.05
    sigma = 0.2
    T = 1.0
    N = 252
    simulations = 1000
    seed = 42

    paths = geometric_brownian_motion(S0, mu, sigma, T, N, simulations, seed)
    analyze_feature_importance(paths, mu, sigma)
