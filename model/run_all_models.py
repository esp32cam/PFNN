# run_all_models.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyts.image import RecurrencePlot
import seaborn as sns
from model_library import get_model
from utils import estimate_lyapunov, multi_embed, compare_rollout_to_true, plot_spectrum, plot_invariant_density

# --- Load your preprocessed data ---
from tvDatafeed import TvDatafeed, Interval
import ta

# Load data
print("Loading data from TradingView...")
tv = TvDatafeed()
df = tv.get_hist('SET', 'SET', interval=Interval.in_daily, n_bars=1200).dropna()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['volatility'] = df['log_return'].rolling(window=10).std()
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['macd'] = ta.trend.MACD(df['close']).macd_diff()
df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

features = ['log_return', 'volume', 'volatility', 'rsi', 'macd', 'adx']
data = df[features].dropna()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- Embed + PCA ---
latent_dim = 18
embedded = multi_embed(data_scaled, delay=1, dimension=3)
latent = PCA(n_components=latent_dim).fit_transform(embedded)
latent_tensor = torch.tensor(latent, dtype=torch.float32)
latent_tensor = (latent_tensor - latent_tensor.mean(0)) / latent_tensor.std(0)

k = int(len(latent_tensor) * 0.3)
Zc, Zm = latent_tensor[:k], latent_tensor[k:]

# --- Run each model ---
model_names = ["pfnn_simple", "koopman_base", "koopman_kan", "koopman_trans", "koopman_trans_svd"]

for model_name in model_names:
    print(f"\n===== Running: {model_name} =====")
    model = get_model(model_name, latent_dim=latent_dim)
    model.eval()

    # Forecast
    z = [latent_tensor[0]]
    for _ in range(k - 1):
        z.append(model(z[-1].unsqueeze(0), mode='contract')[0].squeeze(0))
    for _ in range(len(latent_tensor) - k):
        z.append(model(z[-1].unsqueeze(0), mode='invariant').squeeze(0))
    z_pred = torch.stack(z).detach().numpy()

    # --- Visualization ---
    lle = estimate_lyapunov(z_pred[:, 0])
    print(f"Estimated LLE: {lle:.4f}")

    plt.figure(figsize=(6, 6))
    rp = RecurrencePlot(threshold='point', percentage=20)
    X_rp = rp.fit_transform(z_pred[:, 0].reshape(1, -1))[0]
    plt.imshow(X_rp, cmap='binary', origin='lower')
    plt.title(f"Recurrence Plot ({model_name})")
    plt.show()

    plot_invariant_density(z_pred, model_name)
    plt.title(f"Invariant Density Map ({model_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

    compare_rollout_to_true(z_pred, latent)
    if hasattr(model, 'Gc') and hasattr(model, 'Gm'):
        plot_spectrum(model.Gc, name=f"Gc ({model_name})")
        plot_spectrum(model.Gm, name=f"Gm ({model_name})")
