import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import mplcursors
from matplotlib.widgets import Slider, Button, RadioButtons

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

# Generate synthetic camera poses
camera_poses = generate_synthetic_data()

# Convert camera poses to PyTorch tensor
ray_origins = torch.FloatTensor(camera_poses)

# Initialize NeRF model
nerf = NeRF()

# Define optimizer for model training
optimizer = torch.optim.Adam(nerf.parameters(), lr=1e-3)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_colors = nerf(ray_origins)
    loss = torch.mean(predicted_colors)
    loss.backward()
    optimizer.step()
    
    # Print training progress
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# Generate colors for a new view
new_ray_origins = torch.FloatTensor([[2.0, 2.0, 2.0]])  # New view origin

with torch.no_grad():
    new_view_colors = nerf(new_ray_origins)

# Normalize predicted colors for visualization
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

# Save the visualization as an image
plt.savefig('interactive_nerf_scene.png', dpi=300, bbox_inches='tight')

# Display the visualization
plt.show()