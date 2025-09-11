#!/bin/bash

### (1) Conda 환경 활성화
# CONDA_BASE=$(conda info --base)
# . "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate difix

source ~/anaconda3/etc/profile.d/conda.sh
conda activate difix

### (2) 라이브러리 경로 추가 (CUDA 등)
export PATH=/home/urp25su_sbpark/anaconda3/bin:${PATH}

export CUDA_VISIBLE_DEVICES=1

export TOKENIZERS_PARALLELISM=false

# ID_FILE=""

PATH_TO_INPUT="/local_data_2/dataset/mugsy_test_black_bg/FTD324_20240129"
PATH_TO_REF="/local_data_2/dataset/mugsy_test_black_bg/FTD324_20240129_ref.png"

OUT_ROOT="/local_data_2/urp25su_sbpark/Difix3D/outputs_multi/videos/FTD324_20240129"

MODEL_PATH="/local_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_sv_prior/model.pkl"

python src/inference_difix_video.py \
        --input_image "$PATH_TO_INPUT" \
        --ref_image "$PATH_TO_REF" \
        --output_dir "$OUT_ROOT" \
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
