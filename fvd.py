import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import ToTensor

def load_video_tensor(path, max_frames=32, resize=(128, 128)):
    cap = cv2.VideoCapture(path)
    frames = []
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {path}")
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        frame_tensor = ToTensor()(frame)
        frames.append(frame_tensor)
        if len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) >= 10:
        return torch.stack(frames, dim=0)  # [T, C, H, W]
    else:
        print(f"[WARN] Skipping {path}, not enough frames")
        return None

def trans(x):
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    return x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

# def calculate_fvd(videos1, videos2, device):
#     from common_metrics_on_video_quality.fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance, get_fvd_logits

#     assert len(videos1) == len(videos2), "Mismatch in video count"
#     x1 = torch.stack(videos1).to(device)
#     x2 = torch.stack(videos2).to(device)

#     x1 = trans(x1)
#     x2 = trans(x2)

#     i3d = load_i3d_pretrained(device=device)
#     f1 = get_fvd_logits(x1, i3d=i3d, device=device)
#     f2 = get_fvd_logits(x2, i3d=i3d, device=device)

#     return frechet_distance(f1, f2)

def calculate_fvd(videos1, videos2, device):
    from common_metrics_on_video_quality.fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance, get_fvd_logits

    assert len(videos1) == len(videos2), "Mismatch in video count"

    # 🔥 Step 1: 모든 영상의 최소 길이로 frame 수를 자름
    lengths = [min(v1.shape[0], v2.shape[0]) for v1, v2 in zip(videos1, videos2)]
    min_len = min(lengths)

    # 🔥 Step 2: 모든 영상을 동일한 frame 수로 자름
    videos1 = [v[:min_len] for v in videos1]
    videos2 = [v[:min_len] for v in videos2]

    # [B, T, C, H, W] → [B, C, T, H, W]
    x1 = torch.stack(videos1).to(device)  # [B, T, C, H, W]
    x2 = torch.stack(videos2).to(device)
    x1 = trans(x1)  # [B, C, T, H, W]
    x2 = trans(x2)

    i3d = load_i3d_pretrained(device=device)

    with torch.no_grad():
        f1 = get_fvd_logits(x1, i3d=i3d, device=device)
        f2 = get_fvd_logits(x2, i3d=i3d, device=device)

    return frechet_distance(f1, f2)


def extract_index_from_filename(filename):
    """Extract index from filename like video_gt_3.mp4 -> 3"""
    basename = os.path.splitext(filename)[0]  # video_gt_3
    tokens = basename.split('_')
    return tokens[-1] if tokens[-1].isdigit() else None

def main(gt_root, pred_root, resize=(128, 128), max_frames=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = sorted(os.listdir(gt_root))

    all_gt, all_pred = [], []
    id_fvd_scores = {}

    for id_name in tqdm(ids, desc="Processing IDs"):
        gt_dir = os.path.join(gt_root, id_name)
        pred_dir = os.path.join(pred_root, id_name)
        if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
            continue

        gt_videos, pred_videos = [], []

        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".mp4")])
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".mp4")])

        pred_map = {extract_index_from_filename(f): f for f in pred_files}

        for gt_file in gt_files:
            gt_path = os.path.join(gt_dir, gt_file)
            index = extract_index_from_filename(gt_file)
            if index is None or index not in pred_map:
                continue

            pred_file = pred_map[index]
            pred_path = os.path.join(pred_dir, pred_file)

            gt_tensor = load_video_tensor(gt_path, max_frames=max_frames, resize=resize)
            pred_tensor = load_video_tensor(pred_path, max_frames=max_frames, resize=resize)

            if gt_tensor is not None and pred_tensor is not None:
                gt_videos.append(gt_tensor)
                pred_videos.append(pred_tensor)

        if len(gt_videos) == 0 or len(pred_videos) == 0:
            continue

        fvd = calculate_fvd(pred_videos, gt_videos, device)
        
        # with open(os.path.join(pred_dir, "fvd.txt"), "w") as f: #
        #     f.write(f"{fvd:.4f}")

        id_fvd_scores[id_name] = fvd
        all_gt.extend(gt_videos)
        all_pred.extend(pred_videos)

    # 전체 FVD 계산
    fvd_total = calculate_fvd(all_pred, all_gt, device) if all_gt and all_pred else None

    print("\n=== Per-ID FVD Scores ===")
    for k, v in id_fvd_scores.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Total FVD ===")
    print(f"FVD (all): {fvd_total:.4f}" if fvd_total is not None else "No valid videos found.")
    
    with open(os.path.join(pred_root, "fvd_total.txt"), "w") as f: #
        f.write("\n=== Total FVD ===\n")
        f.write(f"FVD (all): {fvd_total:.4f}\n" if fvd_total is not None else "No valid videos found.")

        
        f.write("\n=== Per-ID FVD Scores ===\n")
        for k, v in id_fvd_scores.items():
            f.write(f"{k}: {v:.4f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, required=True, help="Path to ground truth videos directory")
    parser.add_argument("--pred_root", type=str, required=True, help="Path to predicted videos directory")
    parser.add_argument("--resize", type=int, nargs=2, default=(128, 128), help="Resize videos to this size")
    parser.add_argument("--max_frames", type=int, default=32, help="Max frames per video")
    args = parser.parse_args()

    main(args.gt_root, args.pred_root, resize=tuple(args.resize), max_frames=args.max_frames)
