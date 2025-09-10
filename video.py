import os
import json
import imageio
import argparse
import numpy as np
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm
from model import Difix
from eval import evaluate  
from PIL import Image

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

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--height', type=int, default=800, help='Height of the input image')
    parser.add_argument('--width', type=int, default=544, help='Width of the input image')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--video', action='store_true', help='If the input is a video')
    parser.add_argument('--frame_rate', type=int, default=16) #

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input images
    if os.path.isdir(args.input_image):
        input_paths = sorted(glob(os.path.join(args.input_image, "*rdr.png")))
        # print(f"Loading input images is successful, #:{len(input_paths)}")
        
        print("Searching in:", os.path.join(args.input_image, "*rdr.png"))
        input_paths = sorted(glob(os.path.join(args.input_image, "*rdr.png")))
        print(f"Found {len(input_paths)} images:")
        
    else:
        input_paths = [args.input_image]


    input_images = []
    output_images = []
    for input_path in input_paths:
        
        input_image = Image.open(input_path).convert('RGB')
        input_images.append(input_image)

    
    # 후처리 및 crop
    for input_image in input_images:
        if isinstance(input_image, torch.Tensor):
            input_image = transforms.ToPILImage()(input_image.cpu() * 0.5 + 0.5)
        input_image = center_crop_to_multiple_of_16(input_image)
        output_images.append(input_image)

    # Save outputs
    if args.video:
        # Save as video
        for i in range(4):
            print("Video is being made\n")
            video_path = os.path.join(args.output_dir, f"video_rdr_{i+1}.mp4")
            writer = imageio.get_writer(video_path, fps=args.frame_rate)
            for output_image in tqdm(output_images[i::4], desc="Saving video"):
                writer.append_data(np.array(output_image))
            writer.close()
            
    