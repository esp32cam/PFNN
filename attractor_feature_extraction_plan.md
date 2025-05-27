# Plan for Attractor Feature Extraction and Integration into ABM

This document outlines the plan for extracting dynamical features (attractor features) from financial time series data and integrating them into the Agent-Based Model (ABM) to influence agent behavior.

## 1. Source Data for Feature Extraction

The primary source data for extracting attractor features will be **interpretable financial time series features**. These will be derived from historical stock data (e.g., daily open, high, low, close, volume). Specifically, we will use features similar to those generated in `train_stock_models.py` *before* any dimensionality reduction (like PCA) or time-delay embedding is applied. These include:

*   **Log Returns:** `log(price_t / price_{t-1})`
*   **Realized Volatility:** Standard deviation of log returns over a defined rolling window.
*   **Trading Volume:** Raw or log-transformed.
*   **Technical Indicators:**
    *   Relative Strength Index (RSI)
    *   Moving Average Convergence Divergence (MACD)
    *   Average Directional Index (ADX)

This selection ensures that the features are directly relatable to market dynamics and financial concepts, making the subsequent attractor features more interpretable. These features will be calculated for each relevant stock ticker.

## 2. Dynamic Mode Decomposition (DMD) Features

DMD helps identify coherent spatio-temporal patterns and their temporal dynamics (frequencies, growth/decay rates) in time series data.

*   **Modification of `train_stock_models.py`:**
    *   The current `train_stock_models.py` script performs DMD analysis but primarily saves plots of eigenvalues and summaries.
    *   It will be **necessary to modify the script to save the raw numerical outputs of the DMD analysis**. This includes:
        *   **DMD Eigenvalues (complex numbers):** For each DMD analysis performed (pre-training, post-model predictions, post-residuals).
        *   **DMD Modes (vectors):** Corresponding to each eigenvalue.
        *   **DMD Dynamics (amplitudes over time):** For each mode.
    *   These numerical values should be saved to disk (e.g., using `.npy` or `.csv` files) in the respective stock and model-specific directories within `stock_analysis_results/`.

*   **Using DMD Features to Characterize Attractors:**
    *   **Dominant Eigenvalues:**
        *   **Magnitude:** Eigenvalues with magnitudes close to 1 indicate persistent, stable modes in the attractor. Magnitudes > 1 suggest unstable, growing dynamics (potential bubbles or sharp transitions), while magnitudes < 1 indicate decaying, transient patterns.
        *   **Angle (Frequency):** The angle of complex eigenvalues reveals the characteristic frequencies or periodicities of dominant oscillations within the attractor.
    *   **Mode Structure:** The DMD modes themselves (vectors in the feature space) represent the specific combinations of financial features (log returns, volatility, RSI, etc.) that co-vary according to the dynamics dictated by the corresponding eigenvalue. Analyzing the structure of dominant modes can help identify recurring market states or regimes.
    *   **Energy/Amplitude of Modes:** The initial amplitude or energy associated with each mode can indicate its importance in representing the overall system dynamics.
    *   These features can collectively describe the attractor's stability, identify key periodicities, and characterize different market regimes or states.

## 3. Lyapunov Exponents

Lyapunov exponents quantify the sensitivity of a dynamical system to initial conditions.

*   **Implementation for Maximal Lyapunov Exponent (MLE):**
    *   A function will be implemented to calculate the MLE from the selected source data (Section 1).
    *   This implementation will likely be based on established algorithms for time series, such as the **Rosenstein algorithm** or the **Kantz algorithm**. These methods typically involve:
        1.  Phase space reconstruction of the input time series (e.g., using time-delay embedding for each individual feature, or for a univariate representation of the system if appropriate).
        2.  Tracking the average rate of divergence of initially nearby trajectories in the reconstructed phase space.
    *   The input to this function will be one of the interpretable time series (e.g., log returns, or perhaps a univariate series derived from the multivariate feature set).

*   **Interpretation and Use:**
    *   **Measure of Chaos:** A positive MLE is a strong indicator of chaotic behavior in the market dynamics, implying sensitive dependence on initial conditions.
    *   **Predictability Horizon:** The magnitude of the MLE is inversely related to the predictability horizon of the system. A larger MLE suggests that long-term forecasting is more challenging.
    *   This feature will help characterize the current level of chaos/predictability in the market.

## 4. Fractal Dimensions

Fractal dimensions measure the complexity or "roughness" of a time series, often reflecting the complexity of the underlying attractor.

*   **Implementation for a Fractal Dimension:**
    *   A function will be implemented to calculate a fractal dimension from the selected source data (Section 1).
    *   A likely candidate is the **Higuchi Fractal Dimension (HFD)** due to its direct applicability to 1D time series without requiring phase space reconstruction and its relative robustness. The algorithm involves calculating the average length of the time series curve at different time scales (`k`).
    *   Alternatively, the **Correlation Dimension** (via the Grassberger-Procaccia algorithm) could be implemented, which would require phase space reconstruction of the source data.

*   **Interpretation and Use:**
    *   **Market Complexity:** A higher fractal dimension generally indicates greater complexity or irregularity in the price movements or feature dynamics.
    *   **Market State/Roughness:** Changes in fractal dimension can signal shifts in market behavior, e.g., from smoother, trending phases to more erratic, noisy phases.
    *   This feature will provide a quantitative measure of the market's current structural complexity.

## 5. Integration into ABM (`basic_stock_market_abm.py`)

The extracted attractor features (DMD outputs, MLE, Fractal Dimension) will be used to inform the ABM dynamics.

*   **Populating `model.attractor_signals`:**
    *   The `basic_stock_market_abm.py` script currently includes a placeholder dictionary: `self.attractor_signals = {"lyapunov_exp": 0.1, "fractal_dim": 1.5}`.
    *   In a live or data-driven setup, this dictionary would be populated dynamically. For each simulation period corresponding to available real data, the attractor features would be calculated from that real data (or from the output of models trained in `train_stock_models.py` if those represent a "filtered" view of the market).
    *   For simulation purposes where external data isn't being fed continuously, these values might be set based on averages from a historical period or sampled from distributions derived from historical calculations.

*   **Modulating Agent Behaviors:**
    *   The values in `model.attractor_signals` will be accessible to all agents in the ABM.
    *   In future development phases, these signals will be used to modulate agent behaviors or decision-making parameters. For example:
        *   A high MLE might increase noise traders' randomness or decrease momentum traders' confidence.
        *   A changing dominant DMD frequency might alter the time windows momentum traders consider.
        *   A low fractal dimension might encourage more deterministic strategies.
    *   This integration aims to make the ABM more adaptive to different market conditions as characterized by these dynamical features, moving beyond static agent rules.

This plan provides a roadmap for enriching the ABM with data-driven insights into the market's dynamical properties, with the goal of creating more realistic and adaptive agent behaviors.
