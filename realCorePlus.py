import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image

# logger
logger.add(
    "debug.log",
    format="{time} {level} {message}",
    level="DEBUG",
    rotation="1 MB",
    compression="zip",
)


# load + preprocess data 

def load_video(video_path):
    logger.info(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise FileNotFoundError(f"Cannot open video file: {video_path}")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    logger.debug(f"Loaded {len(frames)} frames from {video_path}")
    return np.array(frames)


def svd_filter(frames, low_cut=1, high_cut=20):
    num_frames, height, width = frames.shape
    casorati = frames.reshape(num_frames, height * width)
    U, Sigma, Vt = np.linalg.svd(casorati, full_matrices=False)
    filtered_sigma = np.zeros_like(Sigma)
    filtered_sigma[low_cut:high_cut] = Sigma[low_cut:high_cut]
    filtered_matrix = U @ np.diag(filtered_sigma) @ Vt
    filtered_frames = filtered_matrix.reshape(num_frames, height, width)
    return np.clip(filtered_frames, 0, 255)


def load_ground_truth(gt_path):
    columns = ["frame_number", "bubble_id", "x", "y", "z", "velocity"]
    return pd.read_csv(gt_path, header=None, names=columns)


import numpy as np
import torch
from torch.utils.data import Dataset


def compute_velocity_components(gt_data, framerate, height, width, pixel_size_x, pixel_size_z):
    grouped = gt_data.groupby("bubble_id")
    velocity_maps = []
    angle_maps = []

    for frame_num in range(int(gt_data["frame_number"].max())):
        vx_map = np.zeros((height, width))
        vz_map = np.zeros((height, width))
        angle_map = np.zeros((height, width))

        for _, group in grouped:
            group = group.sort_values("frame_number")
            for i in range(len(group) - 1):
                f1, f2 = group.iloc[i], group.iloc[i + 1]

                if f2["frame_number"] - f1["frame_number"] == 1 and int(f1["frame_number"]) == frame_num:
                    dx = (f2["x"] - f1["x"]) * 1000
                    dz = (f2["z"] - f1["z"]) * 1000
                    dt = 1 / framerate

                    vx = dx / dt
                    vz = dz / dt
                    angle = np.arctan2(vz, vx)

                    x_pixel = int((f1["x"] * 1000 + 15) / pixel_size_x)
                    z_pixel = int((f1["z"] * 1000 - 47) / pixel_size_z)

                    if 0 <= x_pixel < width and 0 <= z_pixel < height:
                        vx_map[z_pixel, x_pixel] = vx
                        vz_map[z_pixel, x_pixel] = vz
                        angle_map[z_pixel, x_pixel] = angle

        velocity_maps.append(np.stack([vx_map, vz_map], axis=0))  
        angle_maps.append(angle_map)  

    return velocity_maps, angle_maps


def compute_average_velocity_and_angle(velocity_maps):
    """
    Compute averaged vx, vz, magnitude, and angle maps from 500 frames.
    """
    vx_all = np.stack([v[0] for v in velocity_maps])  
    vz_all = np.stack([v[1] for v in velocity_maps])  

    vx_mean = np.mean(vx_all, axis=0)
    vz_mean = np.mean(vz_all, axis=0)

    magnitude = np.sqrt(vx_mean**2 + vz_mean**2)
    angle = np.arctan2(vz_mean, vx_mean)

    return vx_mean, vz_mean, magnitude, angle


# dataset class

class UltrasoundImageDataset(Dataset):
    def __init__(self, vel_dir, angle_dir, seq_len=16):
        self.seq_len = seq_len
        self.vel_paths = sorted([os.path.join(vel_dir, f) for f in os.listdir(vel_dir) if f.endswith('.png')])
        self.ang_paths = sorted([os.path.join(angle_dir, f) for f in os.listdir(angle_dir) if f.endswith('.png')])
        assert len(self.vel_paths) == len(self.ang_paths), "Mismatch in number of velocity and angle maps"
        self.length = len(self.vel_paths) - seq_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load sequence of grayscale velocity images
        input_seq = []
        for i in range(self.seq_len):
            vel_img = np.array(Image.open(self.vel_paths[idx + i]).convert('L'), dtype=np.float32) / 255.0
            input_seq.append(vel_img[None, :, :])  

        input_tensor = torch.from_numpy(np.stack(input_seq)).float()  

        # load target maps
        vel_target = np.array(Image.open(self.vel_paths[idx + self.seq_len - 1]).convert('L'), dtype=np.float32) / 255.0
        ang_target = np.array(Image.open(self.ang_paths[idx + self.seq_len - 1]).convert('L'), dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(np.stack([vel_target, ang_target])).float()  

        return input_tensor, target_tensor


#  model 

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class UNet_ConvLSTM(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Conv2d(n_channels, hidden_dim, 3, padding=1)
        self.lstm = ConvLSTMCell(hidden_dim, hidden_dim, 3)
        self.decoder = nn.Conv2d(hidden_dim, n_classes, 3, padding=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h_t = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        c_t = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        for t in range(T):
            x_t = self.encoder(x[:, t])
            h_t, c_t = self.lstm(x_t, (h_t, c_t))
        return self.decoder(h_t)


# utility

def create_structural_map(preds, frames):
    h, w = frames.shape[1:]
    structural_map = np.zeros((h, w))
    for pred in preds:
        magnitude = np.sqrt(pred[0] ** 2 + pred[1] ** 2)
        structural_map += magnitude
    return gaussian_filter(structural_map / np.max(structural_map), sigma=1)


def visualize_structural_map(structural_map, mean_bmode, output_path="structural_map.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_bmode, cmap="gray", alpha=0.5)
    plt.imshow(structural_map, cmap="jet", alpha=0.7)
    plt.colorbar(label="Flow Intensity")
    plt.title("Super-Resolved Structural Map")
    plt.axis("off")
    plt.savefig(output_path)
    plt.close()
