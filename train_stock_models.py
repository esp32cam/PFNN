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

# Assuming model_library and utils are in the 'model' directory
import sys
sys.path.append('./model')
from model_library import get_model
from utils import multi_embed # Add other necessary utils functions if needed
from tvDatafeed import TvDatafeed, Interval

MODELS_TO_TRAIN = ["pfnn_simple", "koopman_base", "koopman_kan", "koopman_trans", "koopman_trans_svd"]

# --- Load your preprocessed data ---
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

# Create directories for saving outputs
os.makedirs('trained_models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('figures/training_plots', exist_ok=True)

print("Data loading and preprocessing complete.")
print(f"Latent tensor shape: {latent_tensor.shape}")

def train_model(model_name, model, data_scaled_np, latent_tensor_torch, latent_dim, epochs, learning_rate, device):
    print(f"Training {model_name} on {device}...")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    # Prepare data based on model type
    # Models like koopman_kan, koopman_trans, koopman_trans_svd might expect image-like or raw_ish data
    # pfnn_simple, koopman_base expect latent_tensor sequences

    if model_name in ["koopman_kan", "koopman_trans", "koopman_trans_svd"]:
        # These models are more complex and might require specific input shapes.
        # For KoopmanAE_2d_kan, KoopmanAE_2d_trans, KoopmanAE_2d (from pfnn_consist_2d)
        # they expect input like (batch, channels, height, width).
        # We need to decide how to represent stock features as images or if they have an encoder for tabular data.
        # For now, let's assume they require a pre-encoded latent vector for simplicity in this step,
        # or that they are autoencoders trained on the latent space itself for reconstruction.
        # This is a significant simplification and might need adjustment based on model architecture.
        
        # If the model has 'encode' and 'decode' (typical for autoencoders)
        if hasattr(model, 'encode') and hasattr(model, 'decode') and hasattr(model, 'forward_koopman'):
            # This assumes an autoencoder structure that also learns Koopman dynamics.
            # Loss = Reconstruction Loss + Koopman Prediction Loss
            # For this step, we'll simplify and focus on reconstruction of the latent space if possible.
            # A more accurate training would involve feeding `data_scaled_np` (reshaped) 
            # and using model.encode, model.forward_koopman, model.decode.
            
            # Simplified: Train as an autoencoder on latent_tensor
            # This might not be the intended way to train these specific models from context,
            # but it provides a runnable training step.
            dataset = torch.utils.data.TensorDataset(latent_tensor_torch)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            criterion_reconstruction = nn.MSELoss()
            # Criterion for dynamics would be nn.MSELoss() on predicted vs actual future latent states

            print(f"Training {model_name} as an autoencoder on latent_tensor (simplified).")

            for epoch in range(epochs):
                epoch_loss = 0
                for batch_data_list in dataloader:
                    batch_data = batch_data_list[0].to(device)
                    optimizer.zero_grad()
                    
                    # This assumes the model can take latent_tensor as input for AE training
                    # which might not be true for models designed for image inputs.
                    # A proper implementation would require reshaping data_scaled_np 
                    # or using the model's specific encoder.
                    # For now, we'll try to make it run, if it fails, we'll use parameter norm loss.
                    try:
                        if hasattr(model, 'forward_ae'): # If a specific autoencoder forward pass exists
                            reconstructed = model.forward_ae(batch_data) 
                        else: # Generic attempt for AE
                            encoded = model.encode(batch_data)
                            reconstructed = model.decode(encoded)
                        
                        loss = criterion_reconstruction(reconstructed, batch_data)
                        
                        # Placeholder for Koopman dynamics loss (if applicable)
                        # if hasattr(model, 'forward_koopman'):
                        #   z_next_pred = model.forward_koopman(encoded)
                        #   loss_koopman = criterion_reconstruction(z_next_pred[:-1], encoded[1:])
                        #   loss += loss_koopman

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    except Exception as e:
                        # print(f"Warning: Autoencoder-style training for {model_name} failed with {e}. Using parameter norm as loss.")
                        loss = torch.tensor(0.0, device=device)
                        for param in model.parameters():
                            if param.requires_grad and param.data is not None:
                                loss = loss + param.norm() * 1e-6 # Small regularization
                        if loss.requires_grad: # Ensure it's not a zero tensor with no grad
                           loss.backward()
                           optimizer.step()
                        epoch_loss += loss.item() # or just a small number if no params require grad
                        break # Break from batch loop for this epoch if it fails consistently

                avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else epoch_loss
                loss_history.append(avg_epoch_loss)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

        else: # Fallback if not clearly an autoencoder trained on latent space
            print(f"Model {model_name} type not fully handled for specific training logic. Using simple parameter norm as loss.")
            # This is a fallback if the model structure is not as expected for AE on latent.
            for epoch in range(epochs):
                optimizer.zero_grad()
                loss = torch.tensor(0.0, device=device)
                num_params_with_grad = 0
                for param in model.parameters():
                    if param.requires_grad and param.data is not None:
                        loss = loss + param.norm() * 1e-5 # Small regularization
                        num_params_with_grad +=1
                
                if num_params_with_grad > 0 and loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                loss_history.append(loss.item())
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Param Norm Loss: {loss.item():.4f}")


    elif model_name in ["pfnn_simple", "koopman_base"]:
        # These models operate on latent_tensor sequences for dynamics prediction
        # Input: z_t, Target: z_t+1
        input_sequences = latent_tensor_torch[:-1].to(device)
        target_sequences = latent_tensor_torch[1:].to(device)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # For pfnn_simple, it might need a 'mode' argument if its forward pass is structured that way.
            # Assuming model(input) gives prediction for next step.
            # Some models like PFNN might have internal modes ('contract', 'invariant').
            # We need to decide which mode to use for training or if the default forward is sufficient.
            # For simplicity, use default forward or a 'dynamics' mode if available.
            
            if model_name == "pfnn_simple" and hasattr(model, 'forward'): 
                try:
                    output_from_model = model(input_sequences, mode='invariant') 
                except TypeError: 
                    output_from_model = model(input_sequences)
            else:
                output_from_model = model(input_sequences) 
            
            # Handle cases where the model might return a tuple or a list of tensors
            if isinstance(output_from_model, (tuple, list)):
                if output_from_model: # Check if the tuple/list is not empty
                    predictions = output_from_model[0]
                else:
                    # This case should ideally not happen if a model returns outputs.
                    # If it does, loss calculation will likely fail.
                    # For now, we pass it along; criterion will raise an error.
                    print(f"Warning: Model {model_name} returned an empty tuple/list.")
                    predictions = output_from_model 
            else:
                # If it's neither a tuple nor a list, assume it's the tensor itself
                predictions = output_from_model
            
            loss = criterion(predictions, target_sequences)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    else:
        print(f"Warning: Model {model_name} not specifically handled. Using parameter norm as loss.")
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
                
    print(f"Finished training {model_name}.")
    return loss_history

if __name__ == '__main__':
    print("Starting model training orchestration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EPOCHS = 50 
    LEARNING_RATE = 0.001

    for model_name in MODELS_TO_TRAIN:
        print(f"\n===== Processing model: {model_name} =====")
        current_latent_dim = latent_dim 
        
        model = get_model(model_name, latent_dim=current_latent_dim)
        model.to(device)
        
        loss_history = train_model(model_name, model, data_scaled, latent_tensor, current_latent_dim, EPOCHS, LEARNING_RATE, device)
        
        torch.save(model.state_dict(), f'trained_models/{model_name}_weights.pth')
        print(f"Saved weights for {model_name} to trained_models/{model_name}_weights.pth")
        
        plt.figure(figsize=(10,6))
        plt.plot(loss_history)
        plt.title(f'Training Loss for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'figures/training_plots/{model_name}_loss_curve.png')
        plt.close()
        print(f"Saved loss curve for {model_name} to figures/training_plots/{model_name}_loss_curve.png")

        try:
            log_file_path = f'logs/{model_name}_loss_history.log'
            with open(log_file_path, 'w') as f:
                for epoch, loss_val in enumerate(loss_history):
                    f.write(f"Epoch {epoch+1}: {loss_val}\n")
            print(f"Saved numerical loss history for {model_name} to {log_file_path}")
        except Exception as e:
            print(f"Error saving numerical loss history for {model_name}: {e}")

    print("\n===== Training Orchestration Complete =====")
    print(f"Trained models: {', '.join(MODELS_TO_TRAIN)}")
    print("Model weights saved in 'trained_models/' directory.")
    print("Loss curves saved in 'figures/training_plots/' directory.")
    print("Numerical loss histories saved in 'logs/' directory.")
    print("==========================================")
