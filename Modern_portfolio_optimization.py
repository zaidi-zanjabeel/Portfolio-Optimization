# MODERN PORTFOLIO THEORY - PORTFOLIO OPTIMIZER

# STEP 1: IMPORT LIBRARIES 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODERN PORTFOLIO THEORY - OPTIMIZER")
print("=" * 60)

# STEP 2: INVESTMENT UNIVERSE 
tickers = {
    'AAPL':  'Apple (Tech Stock)',
    'MSFT':  'Microsoft (Tech Stock)',
    'JPM':   'JPMorgan (Banking)',
    'JNJ':   'Johnson & Johnson (Healthcare)',
    'GLD':   'Gold ETF',
    'TLT':   'US Bond ETF (20yr Treasury)',
    'VNQ':   'Real Estate ETF (REIT)',
    'SPY':   'S&P 500 Index ETF'
}

symbols = list(tickers.keys())
names   = list(tickers.values())

print("\n Step 2: Downloading stock data from Yahoo Finance...")
print("Assets chosen:")
for sym, name in tickers.items():
    print(f"{sym:6s} → {name}")

raw = yf.download(symbols, start="2019-01-01", end="2024-01-01",
                  auto_adjust=True, progress=False)

prices = raw['Close'].dropna()
print(f"\n Data downloaded! Shape: {prices.shape}")
print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")

# STEP 3: CALCULATION OF RETURN & RISK
print("\n Step 3: Calculating returns, risk & correlations...")

daily_returns = prices.pct_change().dropna()
annual_returns = daily_returns.mean() * 252
annual_risk = daily_returns.std() * np.sqrt(252)
cov_matrix = daily_returns.cov() * 252
corr_matrix = daily_returns.corr()

print("\n Individual Asset Performance:")
print(f"{'Asset':<6} {'Return':>10} {'Risk (Vol)':>12} {'Name'}")
print("   " + "-" * 55)
for i, sym in enumerate(symbols):
    r = annual_returns[sym] * 100
    v = annual_risk[sym] * 100
    print(f"{sym:<6} {r:>9.1f}%  {v:>10.1f}%   {names[i]}")

# STEP 4: MONTE CARLO SIMULATION - 10,000 RANDOM PORTFOLIOS
print("\n Step 4: Running Monte Carlo Simulation (10,000 portfolios)...")

NUM_PORTFOLIOS = 10000
num_assets     = len(symbols)
RISK_FREE_RATE = 0.05  

port_returns   = np.zeros(NUM_PORTFOLIOS)
port_risks     = np.zeros(NUM_PORTFOLIOS)
port_sharpe    = np.zeros(NUM_PORTFOLIOS)
port_weights   = np.zeros((NUM_PORTFOLIOS, num_assets))

np.random.seed(42)  

for i in range(NUM_PORTFOLIOS):
    w = np.random.random(num_assets)
    w = w / w.sum()
    port_weights[i] = w

    ret = np.dot(w, annual_returns.values)
    port_returns[i] = ret
    risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))
    port_risks[i] = risk

    port_sharpe[i] = (ret - RISK_FREE_RATE) / risk

print(f"{NUM_PORTFOLIOS:,} portfolios simulated!")

# STEP 5: SCIPY OPTIMIZATION - TRUE OPTIMAL PORTFOLIOS 
print("\n Step 5: Finding true optimal portfolios via optimization...")

def portfolio_return(weights):
    return np.dot(weights, annual_returns.values)

def portfolio_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

def neg_sharpe(weights):
    r   = portfolio_return(weights)
    s   = portfolio_risk(weights)
    return -(r - RISK_FREE_RATE) / s  

constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

bounds = tuple((0.0, 1.0) for _ in range(num_assets))

w0 = np.array([1.0 / num_assets] * num_assets)

# MAX SHARPE RATIO PORTFOLIO (Best Risk-Adjusted Return)
result_sharpe = minimize(neg_sharpe, w0,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
w_sharpe = result_sharpe.x
ret_sharpe  = portfolio_return(w_sharpe)
risk_sharpe = portfolio_risk(w_sharpe)
sr_sharpe   = (ret_sharpe - RISK_FREE_RATE) / risk_sharpe

# MIN VARIANCE PORTFOLIO (Safest Portfolio)
def port_var(weights):
    return np.dot(weights.T, np.dot(cov_matrix.values, weights))

result_minvar = minimize(port_var, w0,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
w_minvar    = result_minvar.x
ret_minvar  = portfolio_return(w_minvar)
risk_minvar = portfolio_risk(w_minvar)

# EFFICIENT FRONTIER CURVE 
target_returns = np.linspace(
    min(annual_returns.values) * 0.8,
    max(annual_returns.values) * 1.1,
    100
)
frontier_risks = []

for target in target_returns:
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, t=target: portfolio_return(w) - t}
    ]
    res = minimize(port_var, w0, method='SLSQP',
                   bounds=bounds, constraints=cons)
    if res.success:
        frontier_risks.append(np.sqrt(res.fun))
    else:
        frontier_risks.append(np.nan)

frontier_risks   = np.array(frontier_risks)
frontier_returns = target_returns

print("Optimization complete!")
print(f"\n Max Sharpe Portfolio  → Return: {ret_sharpe*100:.1f}%  Risk: {risk_sharpe*100:.1f}%  Sharpe: {sr_sharpe:.2f}")
print(f" Min Variance Portfolio → Return: {ret_minvar*100:.1f}%  Risk: {risk_minvar*100:.1f}%")

# STEP 6: CHARTS 
print("\n Step 6: Creating charts...")

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Modern Portfolio Theory - Portfolio Optimization',
             fontsize=16, fontweight='bold', y=0.98)

# CHART 1: EFFICIENT FRONTIER 
ax1 = axes[0]
sc = ax1.scatter(port_risks * 100, port_returns * 100,
                 c=port_sharpe, cmap='viridis',
                 alpha=0.4, s=8, label='Random Portfolios')
plt.colorbar(sc, ax=ax1, label='Sharpe Ratio')

# Efficient Frontier line
valid = ~np.isnan(frontier_risks)
ax1.plot(frontier_risks[valid] * 100,
         frontier_returns[valid] * 100,
         'r-', linewidth=2.5, label='Efficient Frontier', zorder=5)

# Max Sharpe point
ax1.scatter(risk_sharpe * 100, ret_sharpe * 100,
            color='gold', s=250, zorder=6, marker='*',
            edgecolors='black', linewidth=0.8,
            label=f'Max Sharpe ({sr_sharpe:.2f})')

# Min Variance point
ax1.scatter(risk_minvar * 100, ret_minvar * 100,
            color='red', s=150, zorder=6, marker='D',
            edgecolors='black', linewidth=0.8,
            label='Min Variance')

ax1.set_xlabel('Risk / Volatility (σ) %', fontsize=11)
ax1.set_ylabel('Expected Annual Return (μ) %', fontsize=11)
ax1.set_title('Efficient Frontier', fontsize=13, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# CHART 2: OPTIMAL WEIGHTS PIE CHART 
ax2 = axes[1]
labels_pie = []
sizes_pie  = []
for i, sym in enumerate(symbols):
    if w_sharpe[i] > 0.01:
        labels_pie.append(f"{sym}\n{w_sharpe[i]*100:.1f}%")
        sizes_pie.append(w_sharpe[i])

colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels_pie)))
wedges, texts = ax2.pie(sizes_pie,
                        labels=labels_pie,
                        colors=colors_pie,
                        startangle=90,
                        wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax2.set_title('Max Sharpe Portfolio\nOptimal Asset Allocation',
              fontsize=13, fontweight='bold')

# CHART 3: INDIVIDUAL ASSET RISK vs RETURN 
ax3 = axes[2]
colors_assets = plt.cm.tab10(np.linspace(0, 1, num_assets))
for i, sym in enumerate(symbols):
    ax3.scatter(annual_risk[sym] * 100,
                annual_returns[sym] * 100,
                color=colors_assets[i], s=120,
                zorder=5, edgecolors='black', linewidth=0.5)
    ax3.annotate(sym,
                 (annual_risk[sym] * 100, annual_returns[sym] * 100),
                 textcoords="offset points", xytext=(6, 4), fontsize=9)

# Plot optimal portfolios on same chart
ax3.scatter(risk_sharpe * 100, ret_sharpe * 100,
            color='gold', s=250, marker='*',
            edgecolors='black', linewidth=0.8,
            zorder=6, label='Max Sharpe Portfolio')
ax3.scatter(risk_minvar * 100, ret_minvar * 100,
            color='red', s=150, marker='D',
            edgecolors='black', linewidth=0.8,
            zorder=6, label='Min Variance Portfolio')

ax3.set_xlabel('Annual Risk / Volatility (σ) %', fontsize=11)
ax3.set_ylabel('Annual Expected Return (μ) %', fontsize=11)
ax3.set_title('Individual Assets:\nRisk vs Return', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Charts saved as 'portfolio_optimization.png'")

# FINAL RESULTS 
print("\n" + "=" * 60)
print("  FINAL RESULTS")
print("=" * 60)

print("\n MAXIMUM SHARPE RATIO PORTFOLIO (Best Overall):")
print(f"Expected Annual Return : {ret_sharpe*100:.2f}%")
print(f"Annual Risk (Std Dev)  : {risk_sharpe*100:.2f}%")
print(f"Sharpe Ratio           : {sr_sharpe:.3f}")
print("\n Asset Weights:")
sharpe_df = pd.DataFrame({
    'Asset': symbols,
    'Name': names,
    'Weight (%)': np.round(w_sharpe * 100, 2)
}).sort_values('Weight (%)', ascending=False)
sharpe_df = sharpe_df[sharpe_df['Weight (%)'] > 0.01]
print(sharpe_df.to_string(index=False))

print("\n  MINIMUM VARIANCE PORTFOLIO (Safest):")
print(f"Expected Annual Return : {ret_minvar*100:.2f}%")
print(f"Annual Risk (Std Dev)  : {risk_minvar*100:.2f}%")
print("\n Asset Weights:")
minvar_df = pd.DataFrame({
    'Asset': symbols,
    'Name': names,
    'Weight (%)': np.round(w_minvar * 100, 2)
}).sort_values('Weight (%)', ascending=False)
minvar_df = minvar_df[minvar_df['Weight (%)'] > 0.01]
print(minvar_df.to_string(index=False))

# Save results to CSV
results_df = pd.DataFrame({
    'Asset':         symbols,
    'Name':          names,
    'Annual Return %': np.round(annual_returns.values * 100, 2),
    'Annual Risk %':   np.round(annual_risk.values * 100, 2),
    'Max Sharpe Weight %':  np.round(w_sharpe * 100, 2),
    'Min Variance Weight %': np.round(w_minvar * 100, 2)
})
results_df.to_csv('portfolio_results.csv', index=False)
print("\n Results saved to 'portfolio_results.csv'")
print("\n ALL DONE! Assignment complete!")
print("=" * 60)
