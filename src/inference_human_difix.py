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
    input_tensors: torch.Tensor,   # [B, C, H, W], float in [0,1]
    ref_tensors: torch.Tensor = None,  # [B, C, H, W] or None
    prompts: list = None,
    height: int = 800,
    width: int = 544,
    multi_view: bool = False,
    batch_size: int = 4,
    save: bool = False,
    output_dir: str = ""
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

    # inference
    outputs = model.sample_batch_multi_tensor(
        image=input_tensors,
        ref_image=ref_tensors,
        batch_size=batch_size
    )  # shape: [B, C, H, W] or list of PIL.Image
    
    if save:
        from torchvision.utils import save_image
        for i, output_image in enumerate(outputs):
            filename = f"output_{i}.png"
            output_path = os.path.join(output_dir, filename)
           
            save_image(output_image, output_path)

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
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
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
    outputs = inference_on_tensors(
        model=model,
        input_tensors=input_tensors[:10],
        ref_tensors=ref_tensors,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save=True
    )

    if isinstance(outputs, torch.Tensor):
        print(f"Output shape: {outputs.shape}")  # [B, C, H, W]
    else:
        print(f"Output list length: {len(outputs)} (type={type(outputs[0])})")
