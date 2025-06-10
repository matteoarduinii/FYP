from realCore import *


def inference_on_test(test_video_path, test_gt_path, model_path, framerate=50, seq_len=16):
    logger.info(f"Starting inference on test video: {test_video_path}")

    frames = load_video(test_video_path)
    filtered = svd_filter(frames)
    height, width = frames.shape[1:]

    gt_data = load_ground_truth(test_gt_path)
    velocity_maps, angle_maps = compute_velocity_components(
        gt_data, framerate, height, width, pixel_size_x=30 / 512, pixel_size_z=60 / 512
    )

    dataset = UltrasoundDataset(velocity_maps, angle_maps, seq_len)
    dataloader = DataLoader(dataset, batch_size=1)

    # load model
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # inference
    preds = []
    with torch.no_grad():
        for seq, _ in dataloader:
            seq = seq.to(device)
            pred = model(seq)
            preds.append(pred.cpu().numpy()[0]) 

    # generate structural map
    mean_bmode = np.mean(filtered, axis=0)
    structural_map = create_structural_map(preds, filtered)
    visualize_structural_map(structural_map, mean_bmode)
    logger.info("Inference complete. Structural map saved.")


if __name__ == "__main__":
    test_video_path = "/rds/general/user/ma1421/home/microbubble_project/data/net_001.avi"
    test_gt_path = "/rds/general/user/ma1421/home/microbubble_project/gt_scat_inside1.txt"
    model_path = "/rds/general/user/ma1421/home/microbubble_project/trained_model50.pth"
    inference_on_test(test_video_path, test_gt_path, model_path)
