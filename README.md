## Hufix

**Hufix: Improving 3D Head Avatar with Difix Models**  

## Quick Start

Simplest command:
```
./inference_quick.sh
```
You just change paths to the specific image and model checkpoint.

command:
```bash
CUDA_VISIBLE_DEVICES=0 python src/inference_difix_video.py \
        --input_image "PATH/TO/INPUT_IMAGE" \
        --ref_image "PATH/TO/REF_IMAGE" \
        --output_dir "PATH/TO/OUTPUT" \
        --model_path "checkpoints/model.pkl" \
        --prompt "remove degradation" \
        --timestep 199 
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
