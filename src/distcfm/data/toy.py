import os
import torch
import lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def get_2d_scaler(data, scaling_type=None):
    if not scaling_type:
        return lambda x: x, lambda x: x
    if scaling_type == "minmax":
        data_min = torch.min(data)
        data_max = torch.max(data)
        data_range = data_max - data_min
        # scale to [-1, 1]
        scaler = lambda x: (x - data_min) / data_range * 2 - 1  
        # inverse scale to original range
        inverse_scaler = lambda x: (x + 1) / 2 * data_range + data_min  
    else:
        raise ValueError(f"Unknown data scaling: {scaling_type}")
    return scaler, inverse_scaler

def generate_8gaussians(n_samples=10000, radius=2.0, std=0.05):
  centers = []
  for i in range(8):
      angle = 2 * np.pi * i / 8
      x = radius * np.cos(angle)
      y = radius * np.sin(angle)
      centers.append((x, y))

  centers = np.array(centers)
  samples = []
  for _ in range(n_samples):
      center = centers[np.random.randint(0, 8)]
      point = np.random.normal(loc=center, scale=std, size=(1, 2))
      samples.append(point)
  return np.vstack(samples)

def generate_gmm(n_samples=10000, n_components=8, means=None, covariances=None):
    ms, cvs = [], []
    
    for i in range(n_components):
        ms.append(np.array(means[i]))
        cvs.append(np.array(covariances[i]))
    
    data = []
    for i in range(n_components):
        data.append(np.random.multivariate_normal(mean=ms[i], cov=cvs[i], size=(n_samples // n_components)))
    data = np.concatenate(data, axis=0)
    return data

def generate_checkerboard(n_samples=10000, num_tiles=4, noise=0.1):
    oversample_factor = 5
    total_needed = int(n_samples * oversample_factor)

    # Generate random points
    x = np.random.rand(total_needed, 2) * num_tiles

    # Filter by checkerboard condition
    mask = ((x[:, 0].astype(int) + x[:, 1].astype(int)) % 2) == 0
    x = x[mask]

    # Trim to exact number
    if len(x) < n_samples:
        raise ValueError("Oversample factor too low — increase it.")
    x = x[:n_samples]

    # Add noise and center
    x += np.random.normal(scale=noise, size=x.shape)
    x -= num_tiles / 2
    return x[:n_samples]

def generate_moons(n_samples=10000, noise=0.1): 
    X, y = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X

def plot_2d_dataset(X, title='2D Dataset', save_path=None):
  """Plot a 2D dataset."""
  plt.figure(figsize=(8, 8))
  plt.scatter(X[:, 0], X[:, 1], s=1)
  plt.title(title)
  plt.axis('equal')
  if save_path:
    plt.savefig(save_path)
  else:
    plt.show()
  plt.close()

def get_2d_dataset(cfg,):
    if cfg.dataset.name == 'concentric_circles':
        X, _ = datasets.make_circles(n_samples=cfg.dataset.n_data_points, noise=cfg.dataset.noise, factor=0.5, random_state=0)
    elif cfg.dataset.name == 'eight_gaussians':
        X = generate_8gaussians(n_samples=cfg.dataset.n_data_points, radius=cfg.dataset.radius, std=cfg.dataset.std)
    elif cfg.dataset.name == 'gmm':
        X = generate_gmm(n_samples=cfg.dataset.n_data_points, n_components=cfg.dataset.components, means=cfg.dataset.means, covariances=cfg.dataset.covariances)
    elif cfg.dataset.name == 'swiss_roll':
        X, _ = datasets.make_swiss_roll(n_samples=cfg.dataset.n_data_points, noise=cfg.dataset.noise, random_state=0)
        X = X[:, [0, 2]] 
    elif cfg.dataset.name == 'checkerboard':
        X = generate_checkerboard(n_samples=cfg.dataset.n_data_points, 
                                num_tiles=cfg.dataset.tiles, noise=cfg.dataset.noise)
    elif cfg.dataset.name == 'moons':
        X = generate_moons(n_samples=cfg.dataset.n_data_points)
    else:
        raise NotImplementedError(
        f'Dataset {cfg.dataset.name} not yet supported.')

    plot_2d_dataset(X, title=cfg.dataset.name, 
                   save_path=os.path.join(cfg.work_dir, f"data.png"))
    return X

class ToyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.trainer.batch_size
        self.n_data_points = cfg.dataset.n_data_points
        self.dataset_name = cfg.dataset.name
        self.cfg = cfg

    def prepare_data(self):
        pass  

    def setup(self, stage=None):
        data = get_2d_dataset(cfg=self.cfg) 
        data = torch.tensor(data, dtype=torch.float32)
        self.scaler, self.inverse_scaler = get_2d_scaler(data, 
                                                         self.cfg.dataset.scaling)
        self.transform = transforms.Compose(
            [
                transforms.Lambda(self.scaler),
            ]
        )
        n_train = int(0.8 * self.cfg.dataset.n_data_points)
        n_val = self.cfg.dataset.n_data_points - n_train
        self.train_data, self.val_data = random_split(data, [n_train, n_val])
        self.test_data = self.val_data  
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, 
                          num_workers=4, pin_memory=True)
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
                          num_workers=4, pin_memory=True)