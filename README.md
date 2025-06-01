# Multi-Stock Agent-Based Market Simulation with PFNN-Driven Dynamics

## Overview

This project presents a scalable, data-driven **Agent-Based Modeling (ABM)** framework for simulating stock market dynamics. By integrating advanced operator-learning models (such as PFNN and Koopman-based methods), the platform enables the modeling of individual stock behavior and emergent market phenomena. The system supports batch simulations, real-time monitoring, and statistical validation against real market data.

---

## Key Features

- **High-Performance ABM Core**: Efficiently simulates hundreds of thousands of agents in parallel using distributed GPU computation (PyTorch DDP/NCCL).
- **Stock-Specific Attractor Dynamics**: Each stock can be driven by learned attractor signals/statistics (from PFNN or similar models), enabling market regimes and chaos to be reflected at the agent level.
- **Statistical Save & Analysis Pipeline**: All simulation outputs—including attractor statistics, Lyapunov exponents, recurrence plots, invariant densities, and prediction scores—are systematically saved for batch analysis and validation.
- **Real-Time Monitoring**: Live dashboards provide immediate insights into price trajectories, volatility, agent profit, and emergent behaviors.
- **Market-Wide Comparative Analysis**: Includes tools for temporal validation, network analysis, trend/volatility comparison, and correlation structure benchmarking against real market data.

---

## Project Results

### Attractor Analysis & Model Integration

- **Multivariate Attractor Visualization**: Embedded price and technical indicator data into latent space using time-delay embedding + PCA, visualizing market attractor topology for each stock.
- **PFNN Prediction & Validation**: Compared predicted attractor trajectories to real ones, using MSE and Hausdorff distance for quantitative assessment.
- **Statistical Diagnostics**: Calculated Largest Lyapunov Exponents, recurrence plots, and invariant density maps, confirming the chaotic or stable nature of each stock's latent dynamics.

### ABM Simulation Outputs

- **Single-Stock and Multi-Stock Simulations**: Ran parallel ABM simulations for S&P500 stocks, each with individualized attractor influence.
- **Scenario Experiments**: Demonstrated the impact of agent count and attractor influence strength on price/volume statistics, highlighting nonlinear effects and regime shifts.
- **Batch Results & Quantitative Analysis**: Aggregated and visualized mean final price/volume as functions of simulation parameters.

### Market Validation

- **Temporal & Volatility Analysis**: Compared rolling correlations, volatility, trend direction, and return distributions between ABM simulations and real market data.
- **Network Structure Comparison**: Benchmarked ABM-generated correlation networks, degree distributions, and clustering statistics against the real market.
- **Prediction Accuracy Assessment**: Evaluated direction accuracy, magnitude correlation, and MAPE for price prediction, identifying strengths and improvement areas.

---

## Best-Practice Recommendations

- **Stock-Specific Attractor Feeding**: For high-fidelity ABM, use attractor statistics and regime clustering from each individual stock to drive agents—this allows for realistic regime switching and adaptive market simulation.
- **Systematic Statistical Validation**: Always benchmark ABM outputs against real market statistics (volatility, correlations, network topology, distributional properties) to calibrate and improve realism.
- **Scalable Experimentation**: Leverage distributed GPU environments and automated batch analysis to systematically explore parameter spaces and scenario impacts.
- **Transparent Result Saving**: Maintain structured storage for all simulation outputs, diagnostic figures, and parameter configurations—this supports reproducibility and meta-analysis.
- **Iterative Model Enhancement**: Use prediction accuracy diagnostics (direction, magnitude, error) as feedback for refining both the operator-learning models and agent decision logic.

---

## Example Figures & Outputs

- Multivariate attractor and density plots (PCA latent space)
- PFNN vs. Real trajectory overlays with quantitative error metrics
- Lyapunov and recurrence diagnostics per stock
- ABM simulation batch plots (price, volume, profit trajectories)
- Market validation dashboards: rolling correlations, volatility ratios, trend R², and network graphs
- Distribution and boxplot comparisons between ABM and real markets

---

## Future Directions

- Enhance agent behavioral heterogeneity (strategy diversity, learning agents)
- Integrate cross-asset and cross-market interaction mechanisms
- Extend to causal inference and stress-testing scenarios (e.g., flash crashes)
- Deploy as a research/teaching toolkit for financial market microstructure analysis

---

*For more details, see the included notebooks and code documentation. Contributions and collaborative inquiries are welcome!*
