# MonteCarlo Stock Predictor 🧠📈

A hybrid stock price forecasting tool that combines **Monte Carlo simulation** with a **Decision Tree Regressor** to enhance predictive insights. This project simulates future stock prices using **Geometric Brownian Motion (GBM)** and leverages machine learning to identify influential variables in the resulting forecast distributions.

## 🔍 Overview

This tool is designed for:
- **Simulating stock price paths** using probabilistic modeling (GBM)
- **Running 1,000+ simulations** to estimate future price distributions
- **Training a Decision Tree model** to analyze and rank variable importance
- Helping users understand potential stock trajectories and key drivers

## 🧰 Tech Stack

- **Language**: Python 3.9+
- **Core Libraries**:
  - `pandas` – data manipulation
  - `numpy` – numerical operations and random generation
  - `matplotlib` – data visualization (optional)
  - `scikit-learn` – Decision Tree Regressor
  - `yfinance` – (optional) for stock data collection

## 📁 Project Structure

```
montecarlo-stock-predictor/
├── data/
│   └── stock_data.csv             # (Optional) Pre-downloaded historical stock data
├── src/
│   ├── gbm_simulator.py           # Simulates price paths using GBM
│   ├── decision_tree_analysis.py  # Trains DecisionTreeRegressor on simulated data
│   ├── utils.py                   # Helper functions
├── main.py                        # Run simulations and analysis
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview
```

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. View simulation results and variable importance rankings.

## 📊 Example Use Case

- Predict the future price path of AAPL stock
- Understand how volatility, drift, and time steps affect projections
- Identify which simulation parameters have the strongest influence on outcome ranges
