import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_synthetic_data(num_samples=50):
    camera_poses = np.random.rand(num_samples, 3) * 10 - 5
    return camera_poses

# NeRF Model
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

# Generate synthetic data
camera_poses = generate_synthetic_data()

# Convert data to PyTorch tensor
ray_origins = torch.FloatTensor(camera_poses)

# Initialize NeRF model
nerf = NeRF()

# Define optimizer
optimizer = torch.optim.Adam(nerf.parameters(), lr=1e-3)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_colors = nerf(ray_origins)
    loss = torch.mean(predicted_colors)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

# Generate new view and visualize
new_ray_origins = torch.FloatTensor([[2.0, 2.0, 2.0]])  # New view origin

with torch.no_grad():
    new_view_colors = nerf(new_ray_origins)

normalized_colors = (predicted_colors - predicted_colors.min()) / (predicted_colors.max() - predicted_colors.min())

# Visualization
plt.scatter(
    ray_origins[:, 0], ray_origins[:, 1],
    c=normalized_colors.detach().numpy(), cmap='viridis'
)
plt.colorbar(label="Normalized Predicted Color")
plt.title("Generated Scene")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
