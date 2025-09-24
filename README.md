## Hufix

**Hufix: Improving 3D Head Avatar with Difix Models**  

## Quick Start

Inference difix by simple command:

```
CUDA_VISIBLE_DEVICES=7 python src/inference_human_difix.py --input_image "/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/BCO039_20240326" \
--ref_image "/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/BCO039_20240326_ref.png" \
--model_path "/node_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_sv_prior/model.pkl"
```

Simplest command:
```
./inference_quick.sh
```
You just change paths to the specific image and model checkpoint.

In inference_quick.sh file,
```
PATH_TO_INPUT="/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/NBP315_20240228"
PATH_TO_REF="/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/NBP315_20240228_ref.png"

OUT_ROOT="/node_data_2/urp25su_sbpark/Difix3D/outputs/"

MODEL_PATH="/node_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_sv_prior/model.pkl"
```

## Using Hufix as a module

You can use Hufix as a module:

```
from model import Difix

model = Difix(
        pretrained_path="/node_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_sv_prior_bk_bg/model.pkl",
        timestep=199,
        mv_unet=True
        )     
model.set_eval()

# You need to change these tensors to your images
input_tensor = torch.rand(2, 3, 802, 550)
ref_tensor = torch.rand(1, 3, 802, 550)


# input: [B, C, H, W], ref: [1(V), C, H, W]
outputs = model.sample_batch_multi_tensor(
        image=input_tensor,
        ref_image=ref_tensor,
        batch_size=4
        )
# output: [B, C, H, W]

```


## Data Preparation

Prepare your dataset in the following JSON format:

```json
{
  "train": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "ref_image": "{PATH_TO_REF_IMAGE}",
      "prompt": "remove degradation"
    }
  },
  "test": {
    "{data_id}": {
      "image": "{PATH_TO_IMAGE}",
      "target_image": "{PATH_TO_TARGET_IMAGE}",
      "ref_image": "{PATH_TO_REF_IMAGE}",
      "prompt": "remove degradation"
    }
  }
}
```

## Manual for Model usage
### Image Preperation

You can use both json and image folders:

1.
```
DATA_DIR/
├── {ID}
│   ├── id{ID}_frame0000_view220700191_rdr.png
│   ├── id{ID}_frame0001_view220700191_gt.png
│   ├── ...
│   └── ...

```
2. Just load json file


### Inference

You can find the details in videos.sh file

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference_difix_video.py \
  --input_image "$PATH_TO_INPUT" \
  --ref_image "$PATH_TO_REF" \
  --output_dir "$PATH_TO_OUTPUT" \
  --model_path "$MODEL_PATH" \
  --prompt "remove degradation" \
  --timestep 199 \
  --video \
  --frame_rate 30 \
  --batch_size 4 \
  --record_time
```
if you use image folders.

and if you are using .json files,

```bash
CUDA_VISIBLE_DEVICES=0 python src/inference_difix.py \
  --json_path "$PATH_TO_JSON" \
  --output_dir "$PATH_TO_OUTPUT" \
  --model_path "$MODEL_PATH" \
  --prompt "remove degradation" \
  --timestep 199 \
  --video \
  --frame_rate 30 \
  --batch_size 4 \
  --record_time
```

And then, you can inference images(videos) by 
```
(bash) .../... $ ./videos.sh
```
