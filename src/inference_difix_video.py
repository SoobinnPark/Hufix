import os
import json
import imageio
import argparse
import numpy as np
import torch
import random #
from PIL import Image
from glob import glob
from tqdm import tqdm
from model import Difix
from eval import evaluate  

from PIL import Image

def batched(iterable, batch_size):
    """Yield batches from iterable."""
    for i in range(0, len(iterable), batch_size):
        end = min(len(iterable), i + batch_size)
        yield iterable[i:end]

def center_crop_to_multiple_of_16(img: Image.Image) -> Image.Image:
    width, height = img.size

    # 16의 배수로 자를 크기 계산
    new_width = width - (width % 16)
    new_height = height - (height % 16)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return img.crop((left, top, right, bottom))

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
    parser.add_argument('--json_path', type=str, default=None, help='Path to the data.json file')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to the reference image or directory')
    # parser.add_argument('--ref_image', action='store_true', help='Whether use or not reference image in inference')
    parser.add_argument('--height', type=int, default=800, help='Height of the input image')
    parser.add_argument('--width', type=int, default=544, help='Width of the input image')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to be used')
    parser.add_argument('--model_name', type=str, default=None, help='Name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--video', action='store_true', help='If the input is a video')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference') # 
    parser.add_argument('--record_time', action='store_true', help='If we want measure inference time') #
    parser.add_argument('--frame_rate', type=int, default=30) #
    parser.add_argument('--multi_view', action='store_true', help='option for multi-reference') #
    parser.add_argument('--view', type=int, default=1) #

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    #################################################################################################
    eval_output_dir = os.path.join(args.output_dir, "eval")
    clean_output_dir = os.path.join(args.output_dir, "clean")
    noisy_output_dir = os.path.join(args.output_dir, "noisy")

    os.makedirs(eval_output_dir, exist_ok=True)
    os.makedirs(clean_output_dir, exist_ok=True)
    os.makedirs(noisy_output_dir, exist_ok=True)
    #################################################################################################

    # Initialize the model
    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=args.model_path,
        timestep=args.timestep,
        mv_unet=True if args.ref_image else False,
        record_time=True if args.record_time else False #
    )
    model.set_eval()
    
    #################################################################################################
    
    if args.json_path:
        input_paths, ref_paths, gt_paths = load_from_json(json_path, split="test")

    else:
        # Load input images
        if os.path.isdir(args.input_image):
            input_paths = sorted(glob(os.path.join(args.input_image, "*rdr.png")))
            # print(f"Loading input images is successful, #:{len(input_paths)}")

            print("Searching in:", os.path.join(args.input_image, "*rdr.png"))
            input_paths = sorted(glob(os.path.join(args.input_image, "*rdr.png")))
            print(f"Found {len(input_paths)} input images:")

        else:
            input_paths = [args.input_image]

        # Load reference images if provided
        if args.ref_image is not None:

            if os.path.isdir(args.ref_image):
                ref_paths = sorted(glob(os.path.join(args.ref_image, "*ref.png")))
                # ref_paths = sorted(glob(os.path.join(args.ref_image, "*frame0000*_gt.png")))
                # ref_paths = ref_paths[:4]
                print(f"Found {len(ref_paths)} ref images:")
            else:
                ref_paths = [args.ref_image]


        # Load and save ground truth images
        gt_paths = sorted(glob(os.path.join(args.ref_image, "*gt.png")))
        print(f"Found {len(gt_paths)} gt images:")

        for target_path in gt_paths: # save in advance
            target_filename = os.path.basename(target_path)
            gt_image = Image.open(target_path)
            gt_image.save(os.path.join(clean_output_dir, target_filename))


    #################################################################################################

    output_images = []
    inference_times = []

    for batch_entries in tqdm(batched(input_paths, args.batch_size), total=len(input_paths) // args.batch_size,  desc="Running inference"): # input_images batch
        input_images = []
        ref_images = []
        prompts = []
    
        for idx, input_path in enumerate(batch_entries):
            
            input_image = Image.open(input_path).convert('RGB')
            input_images.append(input_image)
            prompts.append(args.prompt)
    
            if args.ref_image:
                if not args.multi_view:
                    # ref_img = Image.open(ref_paths[(idx + 1) % 4]).convert('RGB') # dv setting 
                    # ref_img = Image.open(ref_paths[(idx) % 4]).convert('RGB') # sv setting
                    ref_img = Image.open(ref_paths[0]) # single video
                    ref_images.append(ref_img)
                else:
                
                    ref_imgs = []

                    for ref_path in ref_paths:
                        ref_img = Image.open(ref_path).convert('RGB') # multi_view
                        ref_imgs.append(ref_img)
                    ref_images.append(ref_imgs)
            
            else:
                ref_images.append(None)

        outputs = model.sample_batch(
            image=input_images,
            height=args.height,
            width=args.width,
            ref_image=ref_images if args.ref_image else None,
            prompt=prompts
        )  # shape: [B, C, H, W] or list of PIL.Image

        if args.record_time:
            elapsed = model.last_inference_time 
            inference_times.append((os.path.basename(input_path), elapsed))


        ##################################################################################
        for input_path, input_image, output_image in zip(batch_entries, input_images, outputs):
            filename = os.path.basename(input_path)
        
            output_path = os.path.join(eval_output_dir, filename)
            output_image.save(output_path)
        
            noisy_save_path = os.path.join(noisy_output_dir, filename)
            input_image.save(noisy_save_path)
        
        ##################################################################################

    
        # 후처리 및 crop
        for output_img in outputs:
            if isinstance(output_img, torch.Tensor):
                output_img = transforms.ToPILImage()(output_img.cpu() * 0.5 + 0.5)
            output_img = center_crop_to_multiple_of_16(output_img)
            output_images.append(output_img)

    # Evaluate
    print("\n=======================\n Results (PSNR / SSIM)...")
    # evaluate(gt_dir=clean_output_dir, pred_dir=eval_output_dir, output_dir=args.output_dir)

    # Save outputs
    if args.video:
        # Save as video
        for i in range(args.view): #
            print("Video is being made", "/n")
            video_path = os.path.join(args.output_dir, f"output_ref_{i+1}.mp4")
            writer = imageio.get_writer(video_path, fps=args.frame_rate)
            for output_image in tqdm(output_images[i::4], desc="Saving video"):
                writer.append_data(np.array(output_image))
            writer.close()
    else:
        # Save as individual images
        for i, output_image in enumerate(tqdm(output_images, desc="Saving images")):
            output_image.save(os.path.join(args.output_dir, os.path.basename(input_images[i])))
    
    # Save timing results
    if args.record_time:
        metrics_path = os.path.join(args.output_dir, "inference_time.txt")
        with open(metrics_path, "w") as f:
            total_time = sum(t for _, t in inference_times)
            avg_time = total_time / len(inference_times)
            f.write(f"# Frame rate: {args.frame_rate}\n\n")
            f.write("# Inference Time Metrics\n")
            f.write(f"Total images: {len(input_paths)}\n")
            f.write(f"Total time: {total_time / 1000:.4f} s\n")
            f.write(f"Average time per image: {avg_time / args.batch_size:.4f} ms\n\n")
            f.write("# Per-image times:\n")
            for fname, t in inference_times:
                f.write(f"{fname}: {t / args.batch_size:.4f} ms\n")

        print(f"\n[✓] Inference time metrics saved to {metrics_path}")
