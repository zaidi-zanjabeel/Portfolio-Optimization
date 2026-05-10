# Modern Portfolio Theory (MPT) Optimizer

## Project Overview
This project is a comprehensive **Portfolio Optimization Tool** based on the principles of Modern Portfolio Theory (MPT). It automates the process of finding the most efficient asset allocations to maximize returns for a given level of risk. The tool bridges the gap between raw market data and actionable investment strategy by providing a clear visual representation of the **Efficient Frontier**.

## Key Features
* **Live Market Data Integration:** Automatically downloads 5+ years of historical data for a diverse universe of assets (Tech, Banking, Healthcare, Gold, Bonds, and REITs) using `yfinance`.
* **Monte Carlo Simulation:** Simulates **10,000 random portfolios** to provide a baseline for the risk-return landscape.
* **Mathematical Optimization:** Uses the `Scipy.optimize` library to calculate:
    * **Maximum Sharpe Ratio Portfolio:** The "optimal" portfolio offering the best risk-adjusted return.
    * **Minimum Variance Portfolio:** The safest possible allocation for risk-averse investors.
* **Efficient Frontier Mapping:** Mathematically calculates and plots the Efficient Frontier curve—the boundary where portfolios are optimized for performance.

## Technical Implementation
* **Optimization Engine:** Employs the `SLSQP` (Sequential Least Squares Programming) method to solve for weights that sum to 100% while staying within established bounds (no short-selling).
* **Risk Metrics:** Calculates annualized returns, volatility (standard deviation), and the covariance matrix to account for asset correlations.
* **Advanced Visualization:** Generates a triple-panel dashboard featuring:
    1. The Efficient Frontier scatter plot.
    2. An optimal asset allocation pie chart.
    3. An individual asset risk-vs-return analysis.

## Portfolio Universe
The model evaluates a diversified mix of assets to ensure robust optimization:
- **Equities:** AAPL, MSFT, JPM, JNJ, SPY
- **Commodities:** GLD (Gold)
- **Fixed Income:** TLT (US Treasury Bonds)
- **Real Estate:** VNQ (REITs)

## How to Use
1. Install dependencies: `pip install numpy pandas matplotlib yfinance scipy`
2. Run the `Modern_portfolio_optimization.py` script.
3. The script will output a detailed CSV report (`portfolio_results.csv`) and save a professional-grade dashboard as `portfolio_optimization.png`.
