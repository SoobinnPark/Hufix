from inference_human_difix import inference_on_tensors
from model import Difix
import torch

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = Difix(
        pretrained_path="/local_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_sv_prior/model.pkl",
        timestep=199,
        mv_unet=True
        ).to(device)
    
    input_tensor = torch.rand(4, 3, 802, 550)
    ref_tensor = torch.rand(1, 3, 802, 550)
    with torch.no_grad():
        outputs = inference_on_tensors(model, input_tensor, ref_tensor)

    print(f"Output shape: {outputs.shape}")
