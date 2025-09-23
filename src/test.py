from inference_human_difix import inference_on_tensors
from model import Difix
import torch

import glob
import os
from torchvision import transforms 
from PIL import Image

from torchvision.utils import save_image

if __name__ == "__main__":
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    
    model = Difix(
        pretrained_path="/node_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_sv_prior_bk_bg/model.pkl",
        timestep=199,
        mv_unet=True
        ) 
    
    model.set_eval()

    # input_tensor = torch.rand(2, 3, 802, 550)
    # ref_tensor = torch.rand(1, 3, 802, 550)

    to_tensor = transforms.ToTensor()

    input_image="/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/BCO039_20240326/BCO039_20240326_frame0000_rdr.png"
    ref_image="/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/BCO039_20240326_ref.png"

    if os.path.isdir(input_image):
        input_files = sorted(glob(os.path.join(input_image, "*rdr.png")))
    else:
        input_files = [input_image]
    
    print(f"Found {len(input_files)} input images")
    print(input_files)
    input_tensor = torch.stack([to_tensor(Image.open(f).convert("RGB")) for f in input_files])
    
    # Load reference images if provided
    ref_tensor = []
    if ref_image is not None:
        if os.path.isdir(ref_image):
            ref_files = sorted(glob(os.path.join(ref_image, "*ref.png")))
        else:
            ref_files = [ref_image]
        print(f"Found {len(ref_files)} reference images")
        ref_tensor = torch.stack([to_tensor(Image.open(f).convert("RGB")) for f in ref_files])

    outputs = model.sample_batch_multi_tensor(
            image=input_tensor,
            ref_image=ref_tensor,
        )

    print(f"Output shape: {outputs.shape}")
