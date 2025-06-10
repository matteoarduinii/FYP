from realCorePlus import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.cuda.amp import GradScaler, autocast


def train_model(train_video_path, velocity_png_dir, angle_png_dir, seq_len=16, epochs=50, batch_size=1):
    logger.info("Starting training process")

    frames = load_video(train_video_path)
    _ = svd_filter(frames)

    # prepare data
    dataset = UltrasoundImageDataset(velocity_png_dir, angle_png_dir, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.warning("CUDA not available. Using CPU.")

    # initialise model
    model = UNet_ConvLSTM(n_channels=1, n_classes=2)
    model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # training loop
    loss_trend = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, gt in dataloader:
            seq, gt = seq.to(device), gt.to(device)

            optimizer.zero_grad()
            with autocast():
                pred = model(seq)
                loss = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * 1000

        avg_loss = total_loss / len(dataloader)
        loss_trend.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Scaled Loss: {avg_loss:.4f}")

    # save model
    torch.save(model.state_dict(), "trained_model_from_png.pth")
    logger.info("Training complete. Model saved as 'trained_model_from_png.pth'")

    # plot loss
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_trend, marker='o')
    plt.title("Training Loss Over Epochs (Scaled by 1000)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE x 1000)")
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    logger.info("Saved training loss plot as 'training_loss_curve.png'")


if __name__ == "__main__":
    train_video_path = "/rds/general/user/ma1421/home/microbubble_project/data/net_002.avi"
    velocity_png_dir = "/rds/general/user/ma1421/home/microbubble_project/data/velocity_outputs$"
    angle_png_dir = "/rds/general/user/ma1421/home/microbubble_project/data/angle_outputs$"
    train_model(train_video_path, velocity_png_dir, angle_png_dir, epochs=50, batch_size=1)
