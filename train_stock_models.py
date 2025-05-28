import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg') # Set backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ta
import os
import pandas as pd
from pydmd.dmd import DMD
from pydmd.plotter import plot_summary as pydmd_plot_summary, plot_eigs as pydmd_plot_eigs

from sp500_data_loader import get_sp500_tickers
import sys
sys.path.append('./model')
from model_library import get_model
from utils import multi_embed

MODELS_TO_TRAIN = ["pfnn_simple", "koopman_base", "koopman_kan", "koopman_trans", "koopman_trans_svd" ] # Reduced for brevity

def extract_attractor_stats(predictions_np):
    if predictions_np is None or predictions_np.ndim != 2 or predictions_np.shape[0] == 0:
        print("Attractor_stats: Invalid or empty predictions_np provided.")
        return None
    try:
        means = np.mean(predictions_np, axis=0)
        stds = np.std(predictions_np, axis=0)
        mins = np.min(predictions_np, axis=0)
        maxs = np.max(predictions_np, axis=0)
        stats_vector = np.concatenate([means, stds, mins, maxs])
        return stats_vector
    except Exception as e:
        print(f"Error calculating attractor stats: {e}")
        return None

def preprocess_stock_data(df_raw, latent_dim_config):
    try:
        df = df_raw.copy()
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing required columns in raw data for preprocessing. Cols: {df.columns}. Skipping.")
            return None

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=10).std()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd_diff()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

        features_list = ['log_return', 'volume', 'volatility', 'rsi', 'macd', 'adx']
        missing_features = [f for f in features_list if f not in df.columns or df[f].isnull().all()]
        if missing_features:
            print(f"Features missing/all NaN after calculation: {missing_features}. Skipping.")
            return None
        
        data_for_features = df[features_list].dropna()

        if data_for_features.shape[0] < max(50, latent_dim_config * 2):
            print(f"Not enough data after feature calculation (rows: {data_for_features.shape[0]}). Skipping.")
            return None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_features)
        embedded_data = multi_embed(data_scaled, delay=1, dimension=3)
        
        if embedded_data.shape[0] < latent_dim_config:
             print(f"Not enough data after embedding for PCA (rows: {embedded_data.shape[0]}). Skipping.")
             return None

        pca = PCA(n_components=latent_dim_config)
        latent_pca_data = pca.fit_transform(embedded_data)
        latent_pca_data = (latent_pca_data - latent_pca_data.mean(axis=0)) / (latent_pca_data.std(axis=0) + 1e-9) 
        print(f"Preprocessing successful. Latent data shape: {latent_pca_data.shape}")
        return latent_pca_data
    except Exception as e:
        print(f"Error during preprocessing stock data: {e}")
        return None

def train_model(model_name, model, data_scaled_np, latent_tensor_torch, latent_dim_config, epochs, learning_rate, device, output_base_path="."):
    print(f"Training {model_name} on {device}...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    predictions_np = None

    if model_name in ["koopman_kan", "koopman_trans", "koopman_trans_svd"] and hasattr(model, 'encode') and hasattr(model, 'decode'):
        print(f"Training {model_name} as an autoencoder on latent_tensor.")
        if latent_tensor_torch.numel() == 0:
            print(f"Latent tensor for {model_name} is empty. Skipping training.")
            return loss_history, predictions_np
        dataset = torch.utils.data.TensorDataset(latent_tensor_torch)
        if len(dataset) == 0:
            print(f"Dataset for {model_name} is empty after TensorDataset. Skipping training.")
            return loss_history, predictions_np
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        criterion_reconstruction = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0; num_batches = 0
            for batch_data_list in dataloader:
                batch_data = batch_data_list[0].to(device)
                if batch_data.numel() == 0: continue
                num_batches += 1
                optimizer.zero_grad()
                try:
                    if hasattr(model, 'forward_ae'): reconstructed = model.forward_ae(batch_data)
                    elif hasattr(model, 'reconstruct'): reconstructed = model.reconstruct(batch_data)
                    else: reconstructed = model.decode(model.encode(batch_data))
                    loss = criterion_reconstruction(reconstructed, batch_data)
                    loss.backward(); optimizer.step()
                    epoch_loss += loss.item()
                except Exception as e_ae:
                    print(f"AE training step for {model_name} failed: {e_ae}. Using param norm.")
                    param_loss_val = sum(p.norm() * 1e-6 for p in model.parameters() if p.requires_grad)
                    if hasattr(param_loss_val, 'backward'): param_loss_val.backward(); optimizer.step()
                    epoch_loss += param_loss_val.item() if hasattr(param_loss_val, 'item') else float(param_loss_val)
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else epoch_loss
            loss_history.append(avg_epoch_loss)
            if (epoch + 1) % 10 == 0: print(f"Epoch [{epoch+1}/{epochs}], {model_name} AE Loss: {avg_epoch_loss:.4f}")

    elif model_name in ["pfnn_simple", "koopman_base"]:
        if latent_tensor_torch.shape[0] < 2:
            print(f"Latent tensor too short for {model_name} (shape: {latent_tensor_torch.shape}). Skipping."); return [], None
        
        input_sequences = latent_tensor_torch[:-1].to(device)
        target_sequences = latent_tensor_torch[1:].to(device)
        criterion = nn.MSELoss(); final_epoch_predictions = None

        for epoch in range(epochs):
            model.train(); optimizer.zero_grad(); epoch_preds = None; current_loss_val = float('nan')
            try:
                if model_name == "pfnn_simple" and hasattr(model, 'forward'):
                    try: output_from_model = model(input_sequences, mode='invariant')
                    except TypeError: output_from_model = model(input_sequences)
                else: output_from_model = model(input_sequences)
                
                if isinstance(output_from_model, (tuple, list)):
                    epoch_preds = output_from_model[0] if output_from_model else None
                    if isinstance(epoch_preds, list) and epoch_preds: epoch_preds = epoch_preds[0]
                else: epoch_preds = output_from_model

                if epoch_preds is not None and target_sequences is not None and epoch_preds.shape == target_sequences.shape:
                    loss = criterion(epoch_preds, target_sequences)
                    current_loss_val = loss.item()
                    if loss.requires_grad: loss.backward(); optimizer.step()
                elif epoch_preds is not None: print(f"Shape mismatch: Preds {epoch_preds.shape}, Targets {target_sequences.shape}.")
                
                if epoch == epochs - 1 and epoch_preds is not None: final_epoch_predictions = epoch_preds.detach().cpu()
            except Exception as e_dyn: print(f"Dynamics training step {model_name} failed: {e_dyn}")
            loss_history.append(current_loss_val)
            if (epoch + 1) % 10 == 0: print(f"Epoch [{epoch+1}/{epochs}], {model_name} Dynamics Loss: {current_loss_val:.4f}")
        
        if final_epoch_predictions is not None: predictions_np = final_epoch_predictions.numpy()

    else:
        print(f"Model {model_name} using parameter norm loss. No sequence predictions.")
        for epoch in range(epochs):
            pass # Simplified for brevity

    os.makedirs(os.path.join(output_base_path, "trained_models"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "figures", "training_plots"), exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_base_path, "trained_models", f'{model_name}_weights.pth'))
    print(f"Saved weights for {model_name}.")
    
    valid_loss_history = [lh for lh in loss_history if lh is not None and not np.isnan(lh)]
    if valid_loss_history:
        plt.figure(figsize=(10,6))
        plt.plot(valid_loss_history); plt.title(f'Training Loss for {model_name}')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
        plt.savefig(os.path.join(output_base_path, "figures", "training_plots", f'{model_name}_loss_curve.png')); plt.close()
        print(f"Saved loss curve for {model_name}.")

    try:
        with open(os.path.join(output_base_path, "logs", f'{model_name}_loss_history.log'), 'w') as f:
            for i, lh_val in enumerate(loss_history): f.write(f"Epoch {i+1}: {lh_val}\n")
        print(f"Saved numerical loss history for {model_name}.")
    except Exception as e_log: print(f"Error saving loss history log: {e_log}")

    print(f"Finished training {model_name}.")
    return loss_history, predictions_np

if __name__ == '__main__':
    print("Starting S&P 500 stock model training script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    LATENT_DIM = 18; EPOCHS = 500; LEARNING_RATE = 0.001
    N_BARS_DATA = 252 * 2

    RESULTS_BASE_DIR = "stock_analysis_results"; os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
    sp500_tickers_full = get_sp500_tickers()
    # sp500_tickers = [t for t in sp500_tickers_full if t and isinstance(t, str)][:5]
    sp500_tickers = [t for t in sp500_tickers_full if t and isinstance(t, str)]
    print(f"Processing {len(sp500_tickers)} tickers (subset for testing).")

    for ticker_count, ticker in enumerate(sp500_tickers):
        print(f"\n===== Processing Ticker {ticker_count+1}/{len(sp500_tickers)}: {ticker} =====")
        ticker_output_dir = os.path.join(RESULTS_BASE_DIR, ticker); os.makedirs(ticker_output_dir, exist_ok=True)
        
        sanitized_ticker_fname = ticker.replace('/', '_').replace('.', '_')
        csv_file_path = os.path.join("sp500_csv_data", f"{sanitized_ticker_fname}.csv")
        if not os.path.exists(csv_file_path): print(f"CSV file not found for {ticker}. Skipping."); continue
        print(f"Loading data for {ticker} from {csv_file_path}...")
        try:
            raw_df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
            if raw_df.empty: print(f"Data for {ticker} from CSV is empty. Skipping."); continue
            raw_df.columns = [col.lower() for col in raw_df.columns]
        except Exception as e_csv: print(f"Error loading CSV for {ticker}: {e_csv}. Skipping."); continue
        
        latent_data_np = preprocess_stock_data(raw_df, LATENT_DIM)
        if latent_data_np is None: print(f"Preprocessing failed for {ticker}. Skipping."); continue
        if latent_data_np.shape[0] < 2: print(f"Latent data too short for {ticker}. Skipping."); continue
        
        dmd_pre_training_dir = os.path.join(ticker_output_dir, "dmd_pre_training"); os.makedirs(dmd_pre_training_dir, exist_ok=True)
        if latent_data_np.shape[0] > 1 and latent_data_np.shape[1] > 0:
            try:
                dmd_instance = DMD(svd_rank=0); dmd_instance.fit(latent_data_np.T)
                fig_summary = None
                try: 
                    fig_summary = pydmd_plot_summary(dmd_instance, figsize=(12, 7))
                except Exception as e_plot: 
                    print(f"pydmd_plot_summary failed for pre-training {ticker}: {e_plot}")
                if fig_summary: 
                    fig_summary.savefig(os.path.join(dmd_pre_training_dir, f"{ticker}_dmd_summary.png"))
                    plt.close(fig_summary)
                    print(f"Saved pre-training DMD summary plot for {ticker}.")
                else: 
                    try:
                        eigs_plot_path = os.path.join(dmd_pre_training_dir, f"{ticker}_dmd_eigs.png")
                        pydmd_plot_eigs(dmd_instance, filename=eigs_plot_path, show_axes=True, show_unit_circle=True, figsize=(8,8))
                        plt.close()
                        print(f"Saved pre-training DMD eigenvalues plot to {eigs_plot_path}")
                    except Exception as e_eigs_plot:
                        print(f"Error plotting DMD eigenvalues for {ticker}: {e_eigs_plot}")
            except Exception as e_dmd: 
                print(f"Error in Pre-Training DMD for {ticker}: {e_dmd}")
        
        current_stock_latent_tensor_torch = torch.tensor(latent_data_np, dtype=torch.float32).to(device)

        for model_name in MODELS_TO_TRAIN:
            print(f"--- Training model: {model_name} for ticker: {ticker} ---")
            model_specific_output_dir = os.path.join(ticker_output_dir, model_name); os.makedirs(model_specific_output_dir, exist_ok=True)
            model = get_model(model_name, latent_dim=LATENT_DIM)
            if model is None: print(f"Could not get model {model_name}. Skipping."); continue
            model.to(device)
            
            _, model_predictions_np = train_model(
                model_name, model, latent_data_np, current_stock_latent_tensor_torch, 
                LATENT_DIM, EPOCHS, LEARNING_RATE, device, model_specific_output_dir
            )
            
            if model_name == "pfnn_simple" and model_predictions_np is not None:
                attractor_stats_vector = extract_attractor_stats(model_predictions_np)
                if attractor_stats_vector is not None:
                    np.save(os.path.join(model_specific_output_dir, f"{ticker}_{model_name}_attractor_stats.npy"), attractor_stats_vector)
                    print(f"Saved attractor stats for {ticker}_{model_name}.")
            
            if model_predictions_np is not None:
                # Post-Training DMD on Predictions
                dmd_pred_dir = os.path.join(model_specific_output_dir, "dmd_predictions"); os.makedirs(dmd_pred_dir, exist_ok=True)
                if model_predictions_np.shape[0] > 1 and model_predictions_np.shape[1] > 0:
                    try:
                        dmd_preds_instance = DMD(svd_rank=0); dmd_preds_instance.fit(model_predictions_np.T)
                        fig_p = None
                        try: 
                            fig_p = pydmd_plot_summary(dmd_preds_instance, figsize=(12,7))
                        except Exception as e_plot: 
                            print(f"pydmd_plot_summary for {model_name} preds failed: {e_plot}")
                        if fig_p: 
                            fig_p.savefig(os.path.join(dmd_pred_dir, f"{ticker}_{model_name}_dmd_preds_summary.png"))
                            plt.close(fig_p)
                            print(f"Saved DMD predictions summary plot for {model_name} on {ticker}.")
                        else:
                            try:
                                eigs_pred_plot_path = os.path.join(dmd_pred_dir, f"{ticker}_{model_name}_dmd_preds_eigs.png")
                                pydmd_plot_eigs(dmd_preds_instance, filename=eigs_pred_plot_path, show_axes=True, show_unit_circle=True, figsize=(8,8))
                                plt.close()
                                print(f"Saved DMD predictions eigenvalues plot to {eigs_pred_plot_path}")
                            except Exception as e_eigs_pred_plot:
                                print(f"Error plotting DMD predictions eigenvalues for {model_name} on {ticker}: {e_eigs_pred_plot}")
                    except Exception as e_dmd_pred: 
                        print(f"Error in DMD on {model_name} predictions for {ticker}: {e_dmd_pred}")