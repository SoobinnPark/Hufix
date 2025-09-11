import os
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from tqdm import tqdm

def evaluate(gt_dir, pred_dir, output_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('_gt.png')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('_rdr.png')])

    psnr_scores = []
    ssim_scores = []
    results_txt = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_metric = FID(feature=2048).to(device)

    # FID에 필요한 리사이징 및 uint8 변환
    transform_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), # PIL Image to Tensor (0-1)
    ])

    for gt_file in tqdm(gt_files, desc="Evaluating"):
        base_name = gt_file.replace("_gt.png", "")
        pred_file = base_name + "_rdr.png"

        gt_path = os.path.join(gt_dir, gt_file)
        pred_path = os.path.join(pred_dir, pred_file)

        if not os.path.exists(pred_path):
            continue

        # 이미지를 한 번만 로드
        try:
            gt_pil = Image.open(gt_path).convert("RGB")
            pred_pil = Image.open(pred_path).convert("RGB")
        except Exception as e:
            print(f"경고: {gt_path} 또는 {pred_path} 이미지 로딩 중 오류 발생. 건너뜁니다. 오류: {e}")
            continue

        # PSNR/SSIM 계산을 위해 numpy 배열로 변환
        gt_np = np.array(gt_pil)
        pred_np = np.array(pred_pil)

        # # --- 추가된 코드: 데이터 범위 및 타입 확인 ---
        # print(f"\n--- 데이터 범위 확인: '{base_name}' ---")
        # print(f"GT (Ground Truth) 이미지: Min={gt_np.min()}, Max={gt_np.max()}, Dtype={gt_np.dtype}")
        # print(f"Pred (Predicted) 이미지: Min={pred_np.min()}, Max={pred_np.max()}, Dtype={pred_np.dtype}")
        # print("---------------------------------------------")
        # # ----------------------------------------------
        
        
        # 데이터 범위를 명시적으로 255로 지정
        psnr = PSNR(gt_np, pred_np, data_range=255)
        ssim = SSIM(gt_np, pred_np, channel_axis=-1, data_range=255)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

        # FID 계산을 위해 299x299로 리사이징하고 uint8 텐서로 변환
        gt_tensor = (transform_fid(gt_pil) * 255).to(torch.uint8).unsqueeze(0).to(device)
        pred_tensor = (transform_fid(pred_pil) * 255).to(torch.uint8).unsqueeze(0).to(device)
        
        fid_metric.update(gt_tensor, real=True)
        fid_metric.update(pred_tensor, real=False)

        result_str = f"{pred_file}; PSNR: {psnr:.2f}, SSIM: {ssim:.4f}"
        results_txt.append(result_str)

    if psnr_scores:
        mean_psnr = np.mean(psnr_scores)
        mean_ssim = np.mean(ssim_scores)
        fid_score = fid_metric.compute().item()

        summary = f"\n========================================\n[Average] PSNR: {mean_psnr:.2f}, SSIM: {mean_ssim:.4f}, FID: {fid_score:.4f}"
        print(summary)
        results_txt.insert(0, summary)

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "metrics.txt")
        with open(save_path, "w") as f:
            f.write("\n".join(results_txt))
        print(f"\n📄 saved to '{save_path}'")
    else:
        print("========================================")
        print("There are no images to evaluate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground truth image directory (with _gt.png)")
    parser.add_argument("--pred_dir", type=str, required=True, help="Predicted output directory (with _rdr.png)")
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    args = parser.parse_args()

    evaluate(args.gt_dir, args.pred_dir, args.output_dir)
