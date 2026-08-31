import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ruamel.yaml import YAML
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from vame.model.rnn_model import RNN_VAE

print(" Initializing Custom PyTorch Pipeline.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware hooked up to: {device}")

# Loading train and test tensors

train_npy=np.load("/its/home/ap2037/dissertation/ssl_vame/vame_pipeline-Aug19-2026/data/train/train_seq.npy")
test_npy=np.load("/its/home/ap2037/dissertation/ssl_vame/vame_pipeline-Aug19-2026/data/train/test_seq.npy")


# Converting to PyTorch Tensors
train_tensor = torch.tensor(train_npy, dtype=torch.float32)
test_tensor = torch.tensor(test_npy, dtype=torch.float32)

# Creating TWO DataLoaders
# We shuffle the training data so it learns general patterns
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)

# We DO NOT shuffle the test data, because we are just evaluating it
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=32, shuffle=False)



# 4. Load your VAME Configuration
# Update this path to your exact project folder
config_path = "/its/home/ap2037/dissertation/ssl_vame/vame_pipeline-Aug19-2026/config.yaml"

yaml =YAML(typ='safe', pure=True)
with open(config_path, 'r') as file:
    config = yaml.load(file)


print("\n Initializing VAME RNN_VAE ")
# Initialize the Neural Network and push it to the GPU
# unpacking the specific hyperparameters to feed the VAME architecture

temporal_window = config['time_window'] *2 # VAME doubles this internally for its bidirectional setup
zdims = config['zdims']
num_features = config['num_features']
future_decoder = config['prediction_decoder']
future_steps = config['prediction_steps']
hidden_size_layer_1 = config['hidden_size_layer_1']
hidden_size_layer_2 = config['hidden_size_layer_2']
hidden_size_rec = config['hidden_size_rec']
hidden_size_pred = config['hidden_size_pred']
dropout_encoder = config['dropout_encoder']
dropout_rec = config['dropout_rec']
dropout_pred = config['dropout_pred']
softplus = config['softplus']

model = RNN_VAE(
    temporal_window, zdims, num_features, future_decoder, future_steps,
    hidden_size_layer_1, hidden_size_layer_2, hidden_size_rec, hidden_size_pred,
    dropout_encoder, dropout_rec, dropout_pred, softplus
).to(device)

# Training
# Optimizer (The mechanism that updates the model's memory)
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

print("\nStarting the Training Loop")

epochs = 100  #
model.train() # 

for epoch in range(epochs):
    total_loss = 0
    
    # The DataLoader automatically scoops the (32, 80, 8) batches
    for batch_idx, batch_data in enumerate(train_loader):
        
        # 
        inputs = batch_data[0].to(device)
        
        # 
        optimizer.zero_grad()
        
        # 
        reconstruction, prediction, latent_space, mu, logvar = model(inputs)
        
        # 
        # standard Mean Squared Error (MSE) loss for this test
        loss = torch.nn.functional.mse_loss(reconstruction, inputs)
        
        # Backward Pass (Calculate the corrections)
        loss.backward()
        
        # Updating the neural network's weights
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

print("\nSuccess! The custom PyTorch training loop has completed.")


print("\n Running Model Evaluation Test ")
model.eval() # model to evaluation mode

# Testing
with torch.no_grad():
    for batch_idx, batch_data in enumerate(test_loader):
        inputs = batch_data[0].to(device)
        
        # Forward pass on the test data
        reconstruction, prediction, latent_space, mu, logvar = model(inputs)
        
        #Future error
        feature_errors = torch.mean((reconstruction - inputs) ** 2, dim=[0, 1]).cpu().numpy()

        # Calculate test loss
        test_loss = torch.nn.functional.mse_loss(reconstruction, inputs)
        print(f"Test Batch {batch_idx + 1} Loss: {test_loss.item():.4f}")
        
        # HEATMAP GENERATION 
        
        # Input shape: [32, 80, 8]  sample shape becomes [80, 8] after indexing
        # Taking the very first sample from this batch (Index 0)
        original_sample = inputs[0].cpu().numpy() 
        reconstructed_target = reconstruction[0].cpu().numpy()

        # 
        body_parts = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3', 'proboscis', 'ovipositor']
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plotting Original Sample (Input Grid)
        im0 = axes[0].imshow(original_sample.T, aspect='auto', cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[0].set_title("Original Sample [0] (Input Contact States)")
        axes[0].set_xlabel("Time Frames (80)")
        axes[0].set_ylabel("Appendages")
        axes[0].set_yticks(ticks=range(8))
        axes[0].set_yticklabels(body_parts)
        
        black_patch = mpatches.Patch(color='black', label='0 (No Contact)')
        # Added a grey edge to the white patch so it's visible against the white background
        white_patch = mpatches.Patch(facecolor='white', edgecolor='lightgray', label='1 (Contact)')
        axes[0].legend(handles=[black_patch, white_patch], loc='lower left', bbox_to_anchor=(0, -0.3), ncol=2)
    
        #Plotting Reconstructed Target (Model Output)
        im1 = axes[1].imshow(reconstructed_target.T, aspect='auto', cmap='gray', interpolation='nearest', vmin=0.0, vmax=1.0)
        axes[1].set_title("Reconstructed Target $\t{x}_t$ (Model Output)")
        axes[1].set_xlabel("Time Frames (80)")
        axes[1].set_yticks(ticks=range(8))
        axes[1].set_yticklabels(body_parts)
        fig.colorbar(im1, ax=axes[1], label='Reconstructed Probability (0.0 to 1.0)')

        plt.suptitle("Visual Comparison: Sample vs. Target Reconstruction", fontsize=14)
        plt.tight_layout()
        # Saving the figure to your outputs folder
        output_path = "/its/home/ap2037/dissertation/ssl_vame/outputs/images/sample_vs_reconstruction.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.clf()
        
        print(f"Comparison heatmap successfully saved to: {output_path}")
        # Plotting the feature errors
        plt.figure(figsize=(10, 5))
        plt.bar(body_parts, feature_errors, color='crimson', alpha=0.8)
        plt.title("Reconstruction Error (MSE) per Appendage")
        plt.xlabel("Appendages / Body Parts")
        plt.ylabel("Mean Squared Error")
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.savefig("/its/home/ap2037/dissertation/ssl_vame/outputs/images/feature_reconstruction_error.png", bbox_inches='tight')
        plt.clf()
        print("Saved feature error bar chart.")
        
        
        
        break