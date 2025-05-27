import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyts.image import RecurrencePlot
import seaborn as sns
import ta
import os
import pandas as pd # Added pandas import
from pydmd.dmd import DMD
from pydmd.plotter import plot_summary as pydmd_plot_summary # Alias to avoid name clashes
# import matplotlib.pyplot as plt # Already imported as plt
# Ensure numpy is imported (already imported as np)
# from sklearn.preprocessing import StandardScaler # Already imported
# from sklearn.decomposition import PCA # Already imported
# Ensure 'ta' (for technical analysis) is imported (already imported)
from sp500_data_loader import get_sp500_tickers # Modified import: fetch_stock_data removed
# from model.utils import multi_embed # This should already be there (it is)


# Assuming model_library and utils are in the 'model' directory
import sys
sys.path.append('./model')
from model_library import get_model
from utils import multi_embed # Add other necessary utils functions if needed
# from tvDatafeed import TvDatafeed, Interval # This will be replaced by yfinance via sp500_data_loader

MODELS_TO_TRAIN = ["pfnn_simple", "koopman_base", "koopman_kan", "koopman_trans", "koopman_trans_svd"]

# Define the new preprocessing function here
def preprocess_stock_data(df_raw, latent_dim_config):
    try:
        df = df_raw.copy()
        # Check for required columns early
        # yfinance data has columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        # We need to adjust to these column names if they are different from 'close', 'high', 'low', 'volume'
        # Assuming yfinance standard column names are already lowercase after fetch_stock_data
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"Missing one or more required columns ({required_cols}) in raw data. Columns found: {df.columns}. Skipping preprocessing.")
            return None

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=10).std()
        
        # Ensure ta is imported if not done globally (already imported globally)
        # import ta 
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd_diff()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

        features_list = ['log_return', 'volume', 'volatility', 'rsi', 'macd', 'adx']
        
        # Ensure all feature columns exist after calculation
        # Some indicators might not be calculable if data is too short
        missing_features = [f for f in features_list if f not in df.columns]
        if missing_features:
            print(f"Missing features after calculation: {missing_features}. Skipping.")
            return None
        
        data_for_features = df[features_list].dropna()

        if data_for_features.shape[0] < max(20, latent_dim_config * 2): # Heuristic: need enough data points
            print(f"Not enough data after feature calculation and dropna for PCA (rows: {data_for_features.shape[0]}). Skipping.")
            return None

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_features)

        # multi_embed is imported from model.utils
        embedded_data = multi_embed(data_scaled, delay=1, dimension=3)
        
        if embedded_data.shape[0] < latent_dim_config:
             print(f"Not enough data after embedding for PCA (rows: {embedded_data.shape[0]}, target latent_dim: {latent_dim_config}). Skipping.")
             return None

        pca = PCA(n_components=latent_dim_config)
        latent_pca_data = pca.fit_transform(embedded_data)
        
        # Normalize latent data
        # Adding epsilon for stability to prevent division by zero if std is zero
        latent_pca_data = (latent_pca_data - latent_pca_data.mean(axis=0)) / (latent_pca_data.std(axis=0) + 1e-9) 

        print(f"Preprocessing successful for stock. Latent data shape: {latent_pca_data.shape}")
        return latent_pca_data
    except Exception as e:
        print(f"Error during preprocessing stock data: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error logging during development
        return None

# --- Old single-stock data loading and preprocessing is now removed/commented ---
# print("Loading data from TradingView...")
# ... (old code for tvDatafeed, feature calculation, PCA, etc.) ...
# os.makedirs('trained_models', exist_ok=True) # Will be handled per ticker
# os.makedirs('logs', exist_ok=True) # Will be handled per ticker
# os.makedirs('figures/training_plots', exist_ok=True) # Will be handled per ticker
# print("Data loading and preprocessing complete.")
# print(f"Latent tensor shape: {latent_tensor.shape}")


# The train_model function definition remains here.
# Minor modifications might be needed later for path handling.
def train_model(model_name, model, data_scaled_np, latent_tensor_torch, latent_dim_config, epochs, learning_rate, device, output_base_path="."):
    print(f"Training {model_name} for a stock on {device}...") # Modified print
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    # Ensure data_scaled_np is not directly used if model expects latent space.
    # The 'latent_tensor_torch' is the primary input for most Koopman models here.
    # 'data_scaled_np' might be needed if a model has its own encoder for raw-ish features,
    # but the current structure seems to train on latent_tensor_torch.
    
    # Initialize predictions_np to None. It will be populated by model types that produce sequence predictions.
    predictions_np = None

    if model_name in ["koopman_kan", "koopman_trans", "koopman_trans_svd"]:
        # These models might be autoencoders operating on the latent space itself,
        # or they might have more complex input requirements.
        # The current simplified approach trains them as AEs on latent_tensor_torch.
        # These AE models, as currently trained, do not produce sequence predictions suitable for this DMD analysis.
        if hasattr(model, 'encode') and hasattr(model, 'decode'): # Basic check for AE structure
            dataset = torch.utils.data.TensorDataset(latent_tensor_torch) # Training AE on latent space
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            criterion_reconstruction = nn.MSELoss()
            print(f"Training {model_name} as an autoencoder on the provided latent_tensor. Sequence predictions for DMD not generated.")

            for epoch in range(epochs):
                epoch_loss = 0
                for batch_data_list in dataloader:
                    batch_data = batch_data_list[0].to(device) # batch_data is a segment of latent_tensor_torch
                    optimizer.zero_grad()
                    try:
                        if hasattr(model, 'forward_ae'):
                            reconstructed = model.forward_ae(batch_data)
                        elif hasattr(model, 'reconstruct'): # Some models might use this name
                            reconstructed = model.reconstruct(batch_data)
                        else: # Generic encode-decode
                            encoded = model.encode(batch_data)
                            reconstructed = model.decode(encoded)
                        
                        loss = criterion_reconstruction(reconstructed, batch_data)
                        
                        # Optional: Add Koopman dynamics loss if model.forward_koopman exists
                        # This requires defining what the Koopman prediction target should be.
                        # For now, focusing on reconstruction of the latent space.

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    except Exception as e:
                        print(f"Warning: AE-style training for {model_name} failed: {e}. Using param norm loss for this batch.")
                        # Fallback to parameter norm loss for this batch if AE forward fails
                        param_loss = torch.tensor(0.0, device=device)
                        for param in model.parameters():
                            if param.requires_grad and param.data is not None:
                                param_loss = param_loss + param.norm() * 1e-6
                        if param_loss.requires_grad:
                           param_loss.backward()
                           optimizer.step()
                        epoch_loss += param_loss.item() # Or a small number if no params require grad
                        # Consider breaking from batch loop if failure is persistent

                avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else epoch_loss
                loss_history.append(avg_epoch_loss)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], AE Loss: {avg_epoch_loss:.4f}")
        else:
            print(f"Model {model_name} does not have standard encode/decode methods for AE training on latent space. Using param norm loss.")
            # Fallback to parameter norm loss if not an AE
            for epoch in range(epochs):
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=device)
                num_params_with_grad = 0
                for param in model.parameters():
                    if param.requires_grad and param.data is not None:
                        loss = loss + param.norm() * 1e-5
                        num_params_with_grad +=1
                if num_params_with_grad > 0 and loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                loss_history.append(loss.item())
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Param Norm Loss: {loss.item():.4f}")

    elif model_name in ["pfnn_simple", "koopman_base"]:
        # Training for dynamics: input z_t, target z_t+1 from latent_tensor_torch
        if latent_tensor_torch.shape[0] < 2:
            print(f"Not enough data points in latent_tensor_torch (shape: {latent_tensor_torch.shape}) to create input/target sequences for {model_name}. Skipping training.")
            # Return loss_history and None for predictions_np
            return loss_history, None 

        input_sequences = latent_tensor_torch[:-1].to(device)
        target_sequences = latent_tensor_torch[1:].to(device)
        criterion = nn.MSELoss()
        
        # Store final predictions from the last epoch for DMD analysis
        final_predictions_tensor = None

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            current_epoch_predictions_tensor = None # To store predictions for this epoch
            if model_name == "pfnn_simple" and hasattr(model, 'forward'): 
                try:
                    output_from_model = model(input_sequences, mode='invariant') 
                except TypeError: 
                    output_from_model = model(input_sequences)
            else:
                output_from_model = model(input_sequences) 
            
            if isinstance(output_from_model, (tuple, list)):
                if output_from_model:
                    current_epoch_predictions_tensor = output_from_model[0]
                    if isinstance(current_epoch_predictions_tensor, list) and current_epoch_predictions_tensor:
                        current_epoch_predictions_tensor = current_epoch_predictions_tensor[0]
                else:
                    print(f"Warning: Model {model_name} returned an empty tuple/list.")
                    current_epoch_predictions_tensor = output_from_model # Will likely cause error in criterion
            else:
                current_epoch_predictions_tensor = output_from_model
            
            # Ensure predictions and targets have compatible shapes
            if current_epoch_predictions_tensor is not None and current_epoch_predictions_tensor.shape != target_sequences.shape:
                print(f"Warning: Shape mismatch for {model_name}. Predictions: {current_epoch_predictions_tensor.shape}, Targets: {target_sequences.shape}. Skipping loss calculation.")
                loss = torch.tensor(float('nan'), device=device) # Or handle error appropriately
            elif current_epoch_predictions_tensor is None:
                print(f"Warning: No predictions generated by {model_name}. Skipping loss calculation.")
                loss = torch.tensor(float('nan'), device=device)
            else:
                loss = criterion(current_epoch_predictions_tensor, target_sequences)
            
            if not torch.isnan(loss) and loss.requires_grad: # Check for NaN and if grad is required
                loss.backward()
                optimizer.step()
            loss_history.append(loss.item())

            if epoch == epochs -1: # Last epoch
                if current_epoch_predictions_tensor is not None:
                    final_predictions_tensor = current_epoch_predictions_tensor.detach().cpu()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Dynamics Loss: {loss.item():.4f}")
        
        if final_predictions_tensor is not None:
            predictions_np = final_predictions_tensor.numpy()

    else: # Fallback for other models or if model_name not in the handled lists
        print(f"Warning: Model {model_name} not specifically handled for dynamics or AE training. Using parameter norm as loss. No sequence predictions generated.")
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            num_params_with_grad = 0
            for param in model.parameters():
                if param.requires_grad and param.data is not None:
                    loss = loss + param.norm() * 1e-5
                    num_params_with_grad +=1
            if num_params_with_grad > 0 and loss.requires_grad:
                loss.backward()
                optimizer.step()
            loss_history.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Param Norm Loss: {loss.item():.4f}")
                
    # Ensure output_base_path subdirectories exist
    model_weights_dir = os.path.join(output_base_path, "trained_models")
    log_dir = os.path.join(output_base_path, "logs")
    plot_dir = os.path.join(output_base_path, "figures/training_plots")
    os.makedirs(model_weights_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Save model weights, plots, logs using output_base_path
    torch.save(model.state_dict(), os.path.join(model_weights_dir, f'{model_name}_weights.pth'))
    print(f"Saved weights for {model_name} to {os.path.join(model_weights_dir, f'{model_name}_weights.pth')}")
    
    if loss_history: # Ensure loss_history is not empty before plotting
        plt.figure(figsize=(10,6))
        plt.plot(loss_history)
        plt.title(f'Training Loss for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{model_name}_loss_curve.png'))
        plt.close()
        print(f"Saved loss curve for {model_name} to {os.path.join(plot_dir, f'{model_name}_loss_curve.png')}")

    try:
        log_file_path = os.path.join(log_dir, f'{model_name}_loss_history.log')
        with open(log_file_path, 'w') as f:
            for epoch, loss_val in enumerate(loss_history): # Ensure loss_history is populated
                f.write(f"Epoch {epoch+1}: {loss_val}\n")
        print(f"Saved numerical loss history for {model_name} to {log_file_path}")
    except Exception as e:
        print(f"Error saving numerical loss history for {model_name}: {e}")

    print(f"Finished training {model_name} for the stock.") # Modified print
    return loss_history, predictions_np # Return predictions_np

if __name__ == '__main__':
    print("Starting S&P 500 stock analysis...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    LATENT_DIM = 18  # Or your desired latent dimension
    EPOCHS = 50      # As previously defined
    LEARNING_RATE = 0.001 # As previously defined
    N_BARS_DATA = 1200 # Number of historical data points (days)

    # MODELS_TO_TRAIN list should be defined (as it was)
    # e.g., MODELS_TO_TRAIN = ["pfnn_simple", "koopman_base", ...]

    RESULTS_BASE_DIR = "stock_analysis_results"
    os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

    sp500_tickers = get_sp500_tickers()
    # For development, you might want to process only a few tickers:
    # sp500_tickers = sp500_tickers[:3] 
    print(f"Found {len(sp500_tickers)} S&P 500 tickers. Processing a subset if specified, otherwise all.")


    for ticker in sp500_tickers:
        print(f"\n===== Processing Ticker: {ticker} =====")
        
        # Create ticker-specific output directory structure
        # This main 'ticker_output_dir' is where all results for this ticker will go.
        # Subdirectories for models, dmd, etc., will be created by respective functions/parts.
        ticker_output_dir = os.path.join(RESULTS_BASE_DIR, ticker)
        os.makedirs(ticker_output_dir, exist_ok=True)
        # Example: model training outputs will go into ticker_output_dir/trained_models/ etc.
        # This 'ticker_output_dir' will be passed to train_model.

        # Load data from CSV
        csv_file_path = os.path.join("sp500_csv_data", f"{ticker}.csv")

        if not os.path.exists(csv_file_path):
            print(f"CSV file not found for {ticker} at {csv_file_path}. Skipping ticker.")
            continue

        print(f"Loading data for {ticker} from {csv_file_path}...")
        try:
            # Assuming the CSV was saved with the date as the first column (index)
            raw_df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
            if raw_df.empty:
                print(f"Data for {ticker} loaded from CSV is empty. Skipping.")
                continue
            # Ensure column names are lowercase, as expected by preprocessing
            raw_df.columns = [col.lower() for col in raw_df.columns]
        except Exception as e:
            print(f"Error loading or parsing CSV for {ticker}: {e}. Skipping ticker.")
            continue
        
        # Pass LATENT_DIM to the preprocessing function
        latent_data_np = preprocess_stock_data(raw_df, LATENT_DIM)
        if latent_data_np is None:
            print(f"Preprocessing failed for {ticker}. Skipping.")
            continue
        
        # Check if latent_data_np has enough rows for sequence based model training
        if latent_data_np.shape[0] < 2: # Minimum 2 for input/target pair
             print(f"Not enough data points in latent_data_np for {ticker} (shape: {latent_data_np.shape}) after preprocessing. Skipping model training for this ticker.")
             continue

        print(f"Successfully preprocessed data for {ticker}. Latent shape: {latent_data_np.shape}")
        
        # --- Pre-Training DMD Analysis ---
        dmd_pre_training_dir = os.path.join(ticker_output_dir, "dmd_pre_training")
        os.makedirs(dmd_pre_training_dir, exist_ok=True)
        print(f"Created directory for pre-training DMD plots: {dmd_pre_training_dir}")

        print(f"Performing Pre-Training DMD for {ticker}...")
        if latent_data_np.shape[0] < 2 or latent_data_np.shape[1] < 1:
            print(f"Skipping DMD for {ticker} due to insufficient data shape: {latent_data_np.shape}")
        else:
            try:
                # PyDMD expects data as (features, snapshots)
                # latent_data_np is (snapshots, features)
                dmd_instance = DMD(svd_rank=0) # svd_rank=0 lets PyDMD try to find optimal truncation
                dmd_instance.fit(latent_data_np.T)

                # Generate and save plots
                fig_summary = None
                try:
                    # Attempt to use pydmd_plot_summary
                    fig_summary = pydmd_plot_summary(dmd_instance, figsize=(12, 7))
                except Exception as e_summary_plot:
                    print(f"Note: pydmd_plot_summary for {ticker} failed or is not supported directly: {e_summary_plot}. Will attempt individual plots.")

                if fig_summary: # If plot_summary returns a figure object
                    summary_plot_path = os.path.join(dmd_pre_training_dir, f"{ticker}_dmd_summary.png")
                    fig_summary.savefig(summary_plot_path)
                    plt.close(fig_summary) # Close the figure to free memory
                    print(f"Saved DMD summary plot to {summary_plot_path}")
                else:
                    # Fallback: If fig_summary is None (either it failed or doesn't return a fig)
                    # Plot eigenvalues
                    try:
                        dmd_instance.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8, 8))
                        eigs_plot_path = os.path.join(dmd_pre_training_dir, f"{ticker}_dmd_eigs.png")
                        plt.savefig(eigs_plot_path)
                        plt.close() # Close the current figure created by plot_eigs
                        print(f"Saved DMD eigenvalues plot to {eigs_plot_path}")
                    except Exception as e_eigs_plot:
                        print(f"Error plotting DMD eigenvalues for {ticker}: {e_eigs_plot}")
                    
                    # Plot modes decomposition (optional, can be verbose)
                    # For example, plot the first mode's dynamics if modes are available
                    # if dmd_instance.modes is not None and dmd_instance.dynamics is not None:
                    #     plt.figure(figsize=(10, 4))
                    #     plt.plot(dmd_instance.dynamics[0, :].real)
                    #     plt.title(f'Dynamics of First DMD Mode for {ticker}')
                    #     mode_dynamics_plot_path = os.path.join(dmd_pre_training_dir, f"{ticker}_dmd_mode1_dynamics.png")
                    #     plt.savefig(mode_dynamics_plot_path)
                    #     plt.close()
                    #     print(f"Saved DMD first mode dynamics plot to {mode_dynamics_plot_path}")
                
                print(f"Finished Pre-Training DMD for {ticker}.")
            except Exception as e:
                print(f"Error during Pre-Training DMD for {ticker}: {e}")
        # --- End of Pre-Training DMD Analysis ---

        current_stock_latent_tensor_torch = torch.tensor(latent_data_np, dtype=torch.float32).to(device)

        # --- Loop through models to train ---
        for model_name in MODELS_TO_TRAIN:
            print(f"--- Training model: {model_name} for ticker: {ticker} ---")
            
            # Create model-specific output directory
            model_specific_output_dir = os.path.join(ticker_output_dir, model_name)
            os.makedirs(model_specific_output_dir, exist_ok=True)
            print(f"Created directory for model-specific outputs: {model_specific_output_dir}")

            # Get model instance
            model = get_model(model_name, latent_dim=LATENT_DIM) 
            if model is None:
                print(f"Could not get model {model_name}. Skipping.")
                continue
            model.to(device)
            
            print(f"Calling train_model for {model_name} on {ticker} data. Latent tensor shape: {current_stock_latent_tensor_torch.shape}")
            # train_model now returns (loss_history, model_predictions_np)
            loss_history, model_predictions_np = train_model(
                model_name=model_name, 
                model=model, 
                data_scaled_np=latent_data_np, # Passing latent_data_np as placeholder for data_scaled_np
                latent_tensor_torch=current_stock_latent_tensor_torch, 
                latent_dim_config=LATENT_DIM, # Parameter name changed in train_model
                epochs=EPOCHS, 
                learning_rate=LEARNING_RATE, 
                device=device,
                output_base_path=model_specific_output_dir # Use model_specific_output_dir
            )
            
            if loss_history: # Check if training even produced a loss history
                 print(f"Loss history for {model_name} on {ticker}: {loss_history[:3 if len(loss_history) > 3 else len(loss_history)]}... (first 3 epochs if available)")
            else:
                print(f"No loss history returned for {model_name} on {ticker}.")

            # --- Post-Training DMD on Predictions ---
            if model_predictions_np is not None:
                dmd_pred_dir = os.path.join(model_specific_output_dir, "dmd_predictions")
                os.makedirs(dmd_pred_dir, exist_ok=True)
                print(f"Performing DMD on Predictions for {model_name} on {ticker}...")
                if model_predictions_np.shape[0] < 2 or model_predictions_np.shape[1] < 1:
                    print(f"Skipping DMD on predictions for {ticker}/{model_name} due to insufficient data shape: {model_predictions_np.shape}")
                else:
                    try:
                        dmd_preds_instance = DMD(svd_rank=0)
                        dmd_preds_instance.fit(model_predictions_np.T) # PyDMD expects (features, snapshots)
                        
                        fig_summary_preds = None
                        try:
                            fig_summary_preds = pydmd_plot_summary(dmd_preds_instance, figsize=(12,7))
                        except Exception as e_plot_summary_preds:
                             print(f"Note: pydmd_plot_summary for predictions of {ticker}/{model_name} failed: {e_plot_summary_preds}. Will attempt individual eigs plot.")

                        if fig_summary_preds:
                            pred_summary_path = os.path.join(dmd_pred_dir, f"{ticker}_{model_name}_dmd_preds_summary.png")
                            fig_summary_preds.savefig(pred_summary_path)
                            plt.close(fig_summary_preds)
                            print(f"Saved DMD on predictions summary to {pred_summary_path}")
                        else: # Fallback to eigs plot
                            dmd_preds_instance.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8,8))
                            eigs_pred_path = os.path.join(dmd_pred_dir, f"{ticker}_{model_name}_dmd_preds_eigs.png")
                            plt.savefig(eigs_pred_path)
                            plt.close()
                            print(f"Saved DMD on predictions eigenvalues plot to {eigs_pred_path}")
                    except Exception as e:
                        print(f"Error during DMD on predictions for {model_name} on {ticker}: {e}")
            else:
                print(f"No predictions returned by {model_name} for {ticker}. Skipping DMD on predictions.")

            # --- Post-Training DMD on Residuals ---
            if model_predictions_np is not None:
                # Predictions from dynamics models (pfnn_simple, koopman_base) are for latent_data_np[1:]
                actuals_for_residuals_np = latent_data_np[1:] 
                
                min_len = min(actuals_for_residuals_np.shape[0], model_predictions_np.shape[0])
                
                if min_len < 2: # Need at least 2 time steps for DMD
                     print(f"Skipping DMD on residuals for {ticker}/{model_name} due to insufficient length after alignment (min_len: {min_len}).")
                else:
                    aligned_actuals = actuals_for_residuals_np[:min_len]
                    aligned_predictions = model_predictions_np[:min_len]
                    
                    residuals_np = aligned_actuals - aligned_predictions
                    
                    dmd_res_dir = os.path.join(model_specific_output_dir, "dmd_residuals")
                    os.makedirs(dmd_res_dir, exist_ok=True)
                    print(f"Performing DMD on Residuals for {model_name} on {ticker}...")
                    if residuals_np.shape[0] < 2 or residuals_np.shape[1] < 1: 
                        print(f"Skipping DMD on residuals for {ticker}/{model_name} due to insufficient data shape after residual calculation: {residuals_np.shape}")
                    else:
                        try:
                            dmd_res_instance = DMD(svd_rank=0)
                            dmd_res_instance.fit(residuals_np.T) # PyDMD expects (features, snapshots)

                            fig_summary_res = None
                            try:
                                fig_summary_res = pydmd_plot_summary(dmd_res_instance, figsize=(12,7))
                            except Exception as e_plot_summary_res:
                                print(f"Note: pydmd_plot_summary for residuals of {ticker}/{model_name} failed: {e_plot_summary_res}. Will attempt individual eigs plot.")

                            if fig_summary_res:
                                res_summary_path = os.path.join(dmd_res_dir, f"{ticker}_{model_name}_dmd_res_summary.png")
                                fig_summary_res.savefig(res_summary_path)
                                plt.close(fig_summary_res)
                                print(f"Saved DMD on residuals summary to {res_summary_path}")
                            else: # Fallback to eigs plot
                                dmd_res_instance.plot_eigs(show_axes=True, show_unit_circle=True, figsize=(8,8))
                                eigs_res_path = os.path.join(dmd_res_dir, f"{ticker}_{model_name}_dmd_res_eigs.png")
                                plt.savefig(eigs_res_path)
                                plt.close()
                                print(f"Saved DMD on residuals eigenvalues plot to {eigs_res_path}")
                        except Exception as e:
                            print(f"Error during DMD on residuals for {model_name} on {ticker}: {e}")
            else:
                print(f"No predictions returned by {model_name} for {ticker}. Skipping DMD on residuals.")
            
            print(f"--- Finished processing model: {model_name} for ticker: {ticker} ---") # Modified print
        
        print(f"===== Finished processing Ticker: {ticker} =====")

    print("\n===== S&P 500 Stock Analysis Complete =====")
    # print(f"Trained models for processed tickers are in subdirectories under '{RESULTS_BASE_DIR}'")
    # print("Model weights, loss curves, and logs are saved per ticker and per model.")
    # print("==========================================")
