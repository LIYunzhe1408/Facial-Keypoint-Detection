import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.image as mpimg

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.index[idx])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, :].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample




class FacialKeypointsHeatmapDataset(Dataset):
    """Face Landmarks dataset with heatmap generation."""

    def __init__(self, csv_file, root_dir, transform=None, output_size=64, sigma=1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            output_size (int): Size of the output heatmaps (default: 64x64)
            sigma (float): Standard deviation for Gaussian kernel (default: 1)
        """
        self.key_pts_frame = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.output_size = output_size
        self.sigma = sigma

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.index[idx])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, :].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}

        if self.transform:
            sample = self.transform(sample)

        # Generate heatmaps
        heatmaps = self.generate_heatmaps(sample["keypoints"])
        sample["heatmaps"] = heatmaps
        
        return sample
        
    def generate_heatmaps(self, keypoints):
      """
      Generate heatmaps for each keypoint by directly computing a 2D Gaussian
      at the keypoint's (x, y) location.

      Args:
          keypoints: Tensor or numpy array of shape (68, 2) for 68 keypoints with (x, y) coordinates

      Returns:
          heatmaps: Tensor of shape (68, output_size, output_size),
                    where each channel is a Gaussian bump for one keypoint.
      """
      # Convert keypoints to numpy if it's a tensor
      if isinstance(keypoints, torch.Tensor):
          keypoints = keypoints.numpy()

      num_keypoints = keypoints.shape[0]
      heatmaps = np.zeros((num_keypoints, self.output_size, self.output_size), dtype=np.float32)

      # Example scaling: from [-1..1] or [0..1] range to your output_size
      # (this line depends on your specific coordinate system)
      keypoints_scaled = keypoints * 50 + 100

      # Create a meshgrid for the pixel coordinates [0..output_size-1]
      # X.shape, Y.shape => (output_size, output_size)
      X, Y = np.meshgrid(np.arange(self.output_size), np.arange(self.output_size))

      for i in range(num_keypoints):
          x_float, y_float = keypoints_scaled[i]

          # Skip if keypoint is invalid
          if np.isnan(x_float) or np.isnan(y_float):
              continue

          # Optionally clamp coordinates to stay in bounds
          x_float = np.clip(x_float, 0, self.output_size - 1)
          y_float = np.clip(y_float, 0, self.output_size - 1)

          # Compute squared distance from each pixel to the keypoint
          dist_sq = (X - x_float)**2 + (Y - y_float)**2

          # 2D Gaussian formula: exp(-dist^2 / (2*sigma^2))
          # self.sigma is your standard deviation
          gaussian = np.exp(-dist_sq / (2 * (self.sigma**2)))

          # If you want to ensure the heatmap sums to 1:
          total = gaussian.sum()
          if total > 0:
              gaussian /= total

          heatmaps[i] = gaussian
          non_zero_mask = (heatmaps != 0)
      return torch.from_numpy(heatmaps)
    
    # def generate_heatmaps(self, keypoints):
    #     """
    #     Generate heatmaps for each keypoint
    #     Args:
    #         keypoints: Tensor or numpy array of shape (68, 2) for 68 keypoints with (x, y) coordinates
    #     Returns:
    #         heatmaps: Tensor of shape (68, output_size, output_size)
    #     """
    #     # Convert keypoints to numpy if it's a tensor
    #     if isinstance(keypoints, torch.Tensor):
    #         keypoints = keypoints.numpy()
        
    #     num_keypoints = keypoints.shape[0]
    #     heatmaps = np.zeros((num_keypoints, self.output_size, self.output_size), dtype=np.float32)
        
       
    #     keypoints_scaled = keypoints * 50 + 100
        
    #     # Generate a heatmap for each keypoint
    #     for i in range(num_keypoints):
    #         # Get the scaled coordinates
    #         x, y = keypoints_scaled[i]
            
    #         # Skip if keypoint is invalid
    #         if np.isnan(x) or np.isnan(y):
    #             continue
            
    #         # Convert to int for indexing
    #         x_int, y_int = max(0, min(self.output_size-1, int(x))), max(0, min(self.output_size-1, int(y)))
            
    #         # Create a single hot pixel
    #         heatmap = np.zeros((self.output_size, self.output_size), dtype=np.float32)
    #         heatmap[y_int, x_int] = 1.0
            
    #         # Apply gaussian filter to create a soft heatmap
    #         heatmap = gaussian_filter(heatmap, sigma=self.sigma)
    #         heatmap = heatmap/heatmap.max()
            
    #         # Normalize heatmap to 0-1 range
    #         if heatmap.max() > 0:
    #             heatmap = heatmap / heatmap.max()
                
    #         heatmaps[i] = heatmap
        
    #     return torch.from_numpy(heatmaps)
