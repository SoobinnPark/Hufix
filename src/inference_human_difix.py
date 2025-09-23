import os
import json
import imageio
import argparse
import numpy as np
import torch
import os
from glob import glob
from PIL import Image
from torchvision import transforms
from model import Difix
from tqdm import tqdm

def batched(iterable, batch_size):
    """Yield batches from iterable."""
    for i in range(0, len(iterable), batch_size):
        end = min(len(iterable), i + batch_size)
        yield iterable[i:end]

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

def inference_on_tensors(
    model,
    input_tensors: torch.Tensor,   # [B, C, H, W], float in [0,1] or [-1,1]
    ref_tensors: torch.Tensor = None,  # [B, C, H, W] or None
    prompts: list = None,
    height: int = 800,
    width: int = 544,
    multi_view: bool = False,
    batch_size: int = 4
):
    """
    Run inference on given input tensors using Difix model.
    
    Args:
        model: Difix
        input_tensors (torch.Tensor): [B, C, H, W]
        ref_tensors (torch.Tensor or None): [V, C, H, W] or None
        prompts (list of str): text prompts, length = B
        height (int): target height
        width (int): target width
        return_pil (bool): if True, return list of PIL.Image, else return torch.Tensor

    Returns:
        outputs: list of PIL.Image or torch.Tensor [B, C, H, W]
    """

    if prompts is None:
        prompts = ["remove degradation"] * batch_size
    
    print(prompts)
    B, C, H, W = input_tensors.shape

    # inference 
    outputs = []
    # print("ref_tensors: ", ref_tensors.shape)

    for batch_entries in tqdm(batched(input_tensors, batch_size), total=len(input_tensors) // batch_size,  desc="Running inference"): # input_tensors batch

        input_batch = batch_entries.unsqueeze(1)
        ref_batch = ref_tensors.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        # print("ref_batch: ", ref_batch.shape)
        output_batch = model.sample_batch_multi_tensor(
            image=input_batch,
            height=height,
            width=width,
            ref_image=ref_batch,
            prompt=prompts[:batch_size]
        )  # shape: [B, C, H, W] or list of PIL.Image

        outputs += output_batch
        
    outputs = torch.stack(outputs, dim=0)  # [B, C, H, W]
    
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument('--json_path', type=str, default=None, help='Path to the data.json file')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to the reference image or directory')
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=544)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    args = parser.parse_args()

    to_tensor = transforms.ToTensor()  # [H,W,C] → [C,H,W] float [0,1]

    if args.json_path:
        input_paths, ref_paths, gt_paths = load_data_from_json(args.json_path, split="test")
        input_tensors = [to_tensor(Image.open(p).convert("RGB")) for p in input_paths]
        ref_tensors   = [to_tensor(Image.open(p).convert("RGB")) for p in ref_paths] if ref_paths else []
        gt_tensors    = [to_tensor(Image.open(p).convert("RGB")) for p in gt_paths] if gt_paths else []

    else:
        # Load input images
        if os.path.isdir(args.input_image):
            input_files = sorted(glob(os.path.join(args.input_image, "*rdr.png")))
        else:
            input_files = [args.input_image]
        
        print(f"Found {len(input_files)} input images")
        input_tensors = torch.stack([to_tensor(Image.open(f).convert("RGB")) for f in input_files])

        # Load reference images if provided
        ref_tensors = []
        if args.ref_image is not None:
            if os.path.isdir(args.ref_image):
                ref_files = sorted(glob(os.path.join(args.ref_image, "*ref.png")))
            else:
                ref_files = [args.ref_image]

            print(f"Found {len(ref_files)} reference images")
            ref_tensors = torch.stack([to_tensor(Image.open(f).convert("RGB")) for f in ref_files])

        # Load ground truth images
        gt_tensors = []
        gt_files = sorted(glob(os.path.join(args.ref_image, "*gt.png"))) if args.ref_image else []
        print(f"Found {len(gt_files)} gt images")

        for f in gt_files:
            img = Image.open(f).convert("RGB")
            gt_tensors.append(to_tensor(img))
        if gt_tensors:
            gt_tensors = torch.stack(gt_tensors)
    
    # Initialize the model
    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=args.model_path,
        timestep=args.timestep,
        mv_unet=True,
    )
    model.set_eval()

    # Enhance the input image
    B = input_tensors.shape[0]
    print(input_tensors.shape)
    outputs = inference_on_tensors(
        model=model,
        input_tensors=input_tensors,
        ref_tensors=ref_tensors,
        prompts=["remove degradation"] * B,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size
    )

    if isinstance(outputs, torch.Tensor):
        print(f"Output shape: {outputs.shape}")  # [B, C, H, W]
    else:
        print(f"Output list length: {len(outputs)} (type={type(outputs[0])})")
