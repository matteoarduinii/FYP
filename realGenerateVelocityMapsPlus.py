from realCorePlus import *
import matplotlib.pyplot as plt
import numpy as np

def normalize_array(arr):
    """Normalize array to [0, 1] for visualization."""
    arr = np.nan_to_num(arr) 
    arr_min, arr_max = np.min(arr), np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def generate_average_velocity_and_angle_png(video_path, gt_path, magnitude_path, angle_path, framerate=50):
    logger.info("Generating average velocity magnitude and angle PNGs...")

    # load video + ground truth
    frames = load_video(video_path)
    gt_data = load_ground_truth(gt_path)
    height, width = frames.shape[1:]

    # compute velocity maps
    velocity_maps, _ = compute_velocity_components(
        gt_data, framerate, height, width,
        pixel_size_x=30 / 512, pixel_size_z=60 / 512
    )

    # compute averaged maps
    _, _, magnitude, angle = compute_average_velocity_and_angle(velocity_maps)

    magnitude_norm = normalize_array(magnitude)
    angle_norm = normalize_array(angle)

    plt.imsave(magnitude_path, magnitude_norm, cmap='jet')
    logger.info(f"Saved average velocity magnitude to {magnitude_path}")

    plt.imsave(angle_path, angle_norm, cmap='hsv')
    logger.info(f"Saved average angle map to {angle_path}")


if __name__ == "__main__":
    video_path = "/rds/general/user/ma1421/home/microbubble_project/data/net_002.avi"
    gt_path = "/rds/general/user/ma1421/home/microbubble_project/data/gt_scat_inside.txt"
    magnitude_output_path = "/rds/general/user/ma1421/home/microbubble_project/velocity_magnitude_avg.png"
    angle_output_path = "/rds/general/user/ma1421/home/microbubble_project/angle_avg.png"

    generate_average_velocity_and_angle_png(
        video_path, gt_path,
        magnitude_output_path, angle_output_path
    )
