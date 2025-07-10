import matplotlib.pyplot as plt
import numpy as np
import torch

from games.block_mesh_splatting.utils.superquadric import implicit_sq

x = np.linspace(-1.5, 1.5, 500)
y = np.linspace(-1.5, 1.5, 50)
z = np.linspace(-1.5, 1.5, 50)
X, Y, Z = np.meshgrid(x, y, z)

points = torch.tensor(np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T, dtype=torch.float32)


eps1 = torch.tensor(1.0)
eps2 = torch.tensor(1.0)
sdf_values = implicit_sq(points, eps1=torch.tensor(0.1), eps2=torch.tensor(1.), as_sdf=True)


sdf_values = sdf_values.reshape(X.shape)

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30, 6))


z_slices = [-1.5, -0.5, 0, 0.5,1.5]
for ax, z_slice in zip(axes, z_slices):
    Z_slice_idx = np.argmin(np.abs(z - z_slice))
    CS = ax.contourf(X[:, :, Z_slice_idx], Y[:, :, Z_slice_idx], sdf_values[:, :, Z_slice_idx], levels=10, cmap='viridis')
    ax.set_title(f'z = {z_slice}')
    fig.colorbar(CS, ax=ax, orientation='vertical')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

plt.show()


