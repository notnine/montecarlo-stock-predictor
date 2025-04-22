
from src.utils import geometric_brownian_motion
from src.decision_tree_analysis import analyze_feature_importance
import matplotlib.pyplot as plt

def explain_and_get_input(prompt, explanation, cast_type=float):
    print(f"\n{prompt}")
    print(f"{explanation}")
    while True:
        try:
            return cast_type(input("Enter value: "))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def run_simulation_and_analysis():
    print("Welcome to the Monte Carlo Stock Predictor!")

    # Explain and get user input for parameters
    S0 = explain_and_get_input(
        "Initial Stock Price (S0):",
        "This is the starting price of the stock. For example, 100 means $100."
    )

    mu = explain_and_get_input(
        "Expected Return / Drift (mu):",
        "This is the average annual return of the stock. Enter as a decimal (e.g., 0.05 for 5%)."
    )

    sigma = explain_and_get_input(
        "Volatility (sigma):",
        "This is how much the stock price tends to fluctuate annually. Enter as a decimal (e.g., 0.2 for 20%)."
    )

    T = explain_and_get_input(
        "Time Horizon (T) in Years:",
        "How far into the future you want to simulate. 1 means one year."
    )

    N = explain_and_get_input(
        "Number of Time Steps (N):",
        "How many steps the simulation should divide the time into. Typically 252 for daily steps in a year.",
        int
    )

    simulations = explain_and_get_input(
        "Number of Simulations:",
        "How many different simulated paths you'd like to generate. More paths = more accuracy.",
        int
    )

    seed = explain_and_get_input(
        "Random Seed (optional, for reproducibility):",
        "This ensures that you get the same random results every time. Enter a number like 42 or leave blank.",
        str
    )

    # Convert empty seed input to None
    seed = int(seed) if seed.strip() != "" else None

    # Run simulation
    paths = geometric_brownian_motion(S0, mu, sigma, T, N, simulations, seed)

    # Plot paths with mean and a synthetic real-world path
    real_world_path = geometric_brownian_motion(S0, mu, sigma, T, N, simulations=1, seed=seed)[0]
    mean_path = paths.mean(axis=0)
    t = range(N)

    plt.figure(figsize=(12, 6))
    for i in range(min(simulations, 100)):
        plt.plot(t, paths[i], color='gray', alpha=0.1)
    plt.plot(t, mean_path, label='Mean Prediction', color='blue', linewidth=2)
    plt.plot(t, real_world_path, label='Real World Path', color='red', linewidth=2)
    plt.title("Monte Carlo Simulated Stock Price Paths vs. Real World")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Analyze and display feature importance
    analyze_feature_importance(paths, mu, sigma)

if __name__ == "__main__":
    run_simulation_and_analysis()
