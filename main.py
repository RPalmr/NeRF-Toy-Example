import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import mplcursors
from matplotlib.widgets import Slider
import os

# Function to generate synthetic camera poses
def generate_synthetic_data(num_samples=50):
    camera_poses = np.random.rand(num_samples, 3) * 10 - 5
    return camera_poses

# Neural Radiance Field (NeRF) Model
class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    
    def forward(self, inputs):
        return self.fc(inputs)

# Training function
def train_nerf(nerf, dataloader, num_epochs=500, learning_rate=1e-3):
    optimizer = torch.optim.Adam(nerf.parameters(), lr=learning_rate)
    loss_history = []  # To store the loss values over epochs
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            predicted_colors = nerf(batch)
            loss = torch.mean(predicted_colors)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)
        
        # Print training progress
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss}")
    
    return loss_history

# Number of training epochs
num_epochs = 500

# Generate synthetic camera poses
camera_poses = generate_synthetic_data()

# Convert camera poses to PyTorch tensor
ray_origins = torch.FloatTensor(camera_poses)

# Initialize NeRF model
nerf = NeRF()

# Create a DataLoader for batched training
batch_size = 16
ray_origins_tensor = torch.tensor(camera_poses, dtype=torch.float32)
dataloader = torch.utils.data.DataLoader(ray_origins_tensor, batch_size=batch_size, shuffle=True)

# Train NeRF model and get loss history
loss_history = train_nerf(nerf, dataloader, num_epochs=num_epochs)

# Normalize predicted colors for visualization
predicted_colors = nerf(ray_origins)
normalized_colors = (predicted_colors - predicted_colors.min()) / (predicted_colors.max() - predicted_colors.min())

# Create a custom colormap for smoother color transitions
initial_colormap = plt.cm.plasma(np.linspace(0, 1, 256))
cmap = LinearSegmentedColormap.from_list("custom_cmap", initial_colormap)

# 3D Scatter Visualization with Smooth Color Gradient and Surface Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    ray_origins[:, 0], ray_origins[:, 1], ray_origins[:, 2],
    c=normalized_colors.detach().numpy(), cmap=cmap, s=50, alpha=0.7
)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Generated Scene with Neural Radiance Field')
ax.set_facecolor('white')

# Add annotations to scatter points
annotations = [f"Color: {color}" for color in cmap(normalized_colors.detach().numpy())]
mplcursors.cursor(scatter, hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(annotations[sel.index])
)

# Add a color bar legend
cbar = plt.colorbar(scatter, ax=ax, pad=0.05, shrink=0.75, aspect=15)
cbar.set_label('Normalized Predicted Color')
cbar.ax.tick_params(labelsize=10)

# Add a slider to customize the colormap
axcolor = 'lightgoldenrodyellow'
ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)
slider = Slider(ax_slider, 'Custom Colormap', 0, 1, valinit=1)

# Function to update colormap based on slider value
def update(val):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", plt.cm.plasma(np.linspace(0, slider.val, 256)))
    scatter.set_cmap(cmap)
    cbar.update_normal(scatter)

slider.on_changed(update)

# Save the visualization as an image in the "images" folder
image_filename = 'interactive_nerf_scene.png'
image_path = os.path.join('images', image_filename)

# Create the "images" folder if it doesn't exist
os.makedirs('images', exist_ok=True)

plt.savefig(image_path, dpi=300, bbox_inches='tight')

# Plot the loss history
plt.figure()
plt.plot(range(num_epochs), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Progress')
plt.grid()
plt.show()

# Display the visualization
plt.show()