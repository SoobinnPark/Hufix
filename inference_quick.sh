#!/bin/bash

### (1) Conda 환경 활성화
# CONDA_BASE=$(conda info --base)
# . "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate difix

source ~/anaconda3/etc/profile.d/conda.sh
conda activate difix

### (2) 라이브러리 경로 추가 (CUDA 등)
export PATH=/home/urp25su_sbpark/anaconda3/bin:${PATH}

export CUDA_VISIBLE_DEVICES=0

export TOKENIZERS_PARALLELISM=false

# ID_FILE=""

PATH_TO_INPUT=""
PATH_TO_REF=""

OUT_ROOT=""

MODEL_PATH=""

python src/inference_difix_video.py \
        --input_image "$ROOT/$ID" \
        --ref_image "$ROOT/$ID" \
        --output_dir "$OUT_ROOT/$ID" \
        --model_path "$MODEL_PATH" \
        --prompt "remove degradation" \
        --timestep 199 \
        --video \
        --frame_rate 30 \
        --batch_size 4 \
        --record_time
        

# for multi-inference usage

# while IFS= read -r ID; do
#     echo "=== Running inference for ID: $ID ==="

#     python src/inference_difix_video_copy.py \
#         --input_image "$ROOT/$ID" \
#         --ref_image "$ROOT/$ID" \
#         --output_dir "$OUT_ROOT/$ID" \
#         --model_path "$MODEL_PATH" \
#         --prompt "remove degradation" \
#         --timestep 199 \
#         --video \
#         --frame_rate 30 \
#         --batch_size 4 \
#         --record_time
        

#     echo ""
# done < "$ID_FILE"