import os
import json
import imageio
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from model import Difix
from eval import evaluate  

def load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    test_entries = []
    gt_paths = []
    for item_id, entry in data.get("test", {}).items():
        input_path = entry["image"]
        target_path = entry["target_image"]
        ref_path = entry.get("ref_image", None)
        prompt = entry.get("prompt", "remove degradation")
        test_entries.append((item_id, input_path, target_path, ref_path, prompt))
        gt_paths.append((item_id, target_path))
    return test_entries, dict(gt_paths)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True, help='Path to the data.json file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the outputs')
    parser.add_argument('--ref_image', type=bool, default=False, help='Whether use or not reference image in inference')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict to be used')
    parser.add_argument('--height', type=int, default=800, help='Height of the input image')
    parser.add_argument('--width', type=int, default=544, help='Width of the input image')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    # parser.add_argument('--video', action='store_true', help='If the input is a video')
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    eval_output_dir = os.path.join(args.output_dir, "eval")
    clean_output_dir = os.path.join(args.output_dir, "clean")
    noisy_output_dir = os.path.join(args.output_dir, "noisy")

    os.makedirs(eval_output_dir, exist_ok=True)
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(noisy_output_dir, exist_ok=True)


    # Initialize model
    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=args.model_path,
        timestep=args.timestep,
        mv_unet=True if args.ref_image else False, #
    )
    model.set_eval()

    # Load data from JSON
    data_entries, gt_path_dict = load_data_from_json(args.json_path)

    print(f"Inference total {len(data_entries)} samples...")

    for item_id, input_path, target_path, ref_path, prompt in tqdm(data_entries, desc="Running inference"):
        # print(input_path, item_id, ref_path, prompt)

        input_image = Image.open(input_path).convert('RGB')
        
        
        ref_image = Image.open(ref_path).convert('RGB') if args.ref_image else None # Load reference images if provided
        # print(ref_image)
        output_image = model.sample(
            input_image,
            height=args.height,
            width=args.width,
            ref_image=ref_image,
            prompt=prompt
        )

        filename = os.path.basename(input_path)
        output_path = os.path.join(eval_output_dir, filename)
        output_image.save(output_path)

        # Save the ground truth image and input image
        target_filename = os.path.basename(target_path)
        target_path = gt_path_dict[item_id]
        gt_image = Image.open(target_path)
        gt_image.save(os.path.join(clean_output_dir, target_filename))

        noisy_save_path = os.path.join(noisy_output_dir, filename)
        input_image.save(noisy_save_path)

    # Evaluate
    print("\n=======================\n Results (PSNR / SSIM)...")
    evaluate(gt_dir=clean_output_dir, pred_dir=eval_output_dir, output_dir=eval_output_dir)
