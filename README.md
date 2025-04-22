# MonteCarlo Stock Predictor 🧠📈

A hybrid stock price forecasting tool that combines **Monte Carlo simulation** with a **Decision Tree Regressor** to enhance predictive insights. This project simulates future stock prices using **Geometric Brownian Motion (GBM)** and leverages machine learning to identify influential variables in the resulting forecast distributions.

## 🔍 Overview

This tool is designed for:
- **Simulating stock price paths** using probabilistic modeling (GBM)
- **Running 1,000+ simulations** to estimate future price distributions
- **Training a Decision Tree model** to analyze and rank variable importance
- Comparing predictions to a synthetic “real world” to assess forecasting behavior

## 🧰 Tech Stack

- **Language**: Python 3.9+
- **Core Libraries**:
  - `pandas` – data manipulation
  - `numpy` – numerical operations and random generation
  - `matplotlib` – data visualization
  - `scikit-learn` – Decision Tree Regressor

## 📁 Project Structure

```
montecarlo-stock-predictor/
├── src/
│   ├── gbm_simulator.py           # Simulates price paths using GBM
│   ├── decision_tree_analysis.py  # Trains DecisionTreeRegressor on simulated data
│   ├── utils.py                   # Helper functions
├── main.py                        # Run simulations and analysis
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview
```

## 📈 Graphs

### 📉 Simulated Forecast vs. Real World
![montecarloexample](https://github.com/user-attachments/assets/c0f69ad1-7ab1-4da3-8b37-c40b616fb3a5)   

This chart overlays:
- **Gray lines**: Predicted simulations from Monte Carlo
- **Blue line**: Mean predicted path
- **Red line**: A synthetic "real world" path to compare against predictions

### 🌲 Feature Importance from Decision Tree
![montecarloexample2](https://github.com/user-attachments/assets/88125f37-43b6-43ef-abc1-4f1a6c6b0591)   

This bar chart ranks:
- How much each input (e.g., drift or volatility) influenced the final simulated stock price
- Helps explain **which variables matter most** in shaping outcomes

## 📊 Visuals

### Forecasted Simulations
A set of 1,000 simulated stock price paths using Geometric Brownian Motion.
- The **gray lines** represent individual simulation paths.
- The **blue line** is the mean of all predictions.

### Real World Path
A single GBM simulation treated as the “actual” future stock price behavior.

### Combined Comparison Visual
An overlay that compares:
- All prediction paths
- The mean predicted path (blue)
- The real-world path (red)

This allows you to visually assess how realistic the model’s forecasts were under synthetic conditions.

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

3. Observe real-time visualizations of prediction vs. synthetic outcome.

## 🧪 Example Use Case

- Predict the future price path of a stock
- Understand how volatility, drift, and time steps affect projections
- Identify key drivers in the forecast and test model accuracy against a known outcome
