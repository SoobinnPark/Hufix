#!/bin/bash

### (1) Conda 환경 활성화
# CONDA_BASE=$(conda info --base)
# . "$CONDA_BASE/etc/profile.d/conda.sh"
# conda activate difix

source ~/anaconda3/etc/profile.d/conda.sh
conda activate difix

### (2) 라이브러리 경로 추가 (CUDA 등)
export PATH=/home/urp25su_sbpark/anaconda3/bin:${PATH}

export CUDA_VISIBLE_DEVICES=2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TOKENIZERS_PARALLELISM=false

# ID_FILE=""

PATH_TO_INPUT="/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/NBP315_20240228"
PATH_TO_REF="/node_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/mugsy_test_black_bg/NBP315_20240228_ref.png"

OUT_ROOT="/node_data_2/urp25su_sbpark/Difix3D/output"

MODEL_PATH="/node_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_sv_prior/model.pkl"

python src/inference_human_difix.py \
        --input_image "$PATH_TO_INPUT" \
        --ref_image "$PATH_TO_REF" \
        --output_dir "$OUT_ROOT" \
        --model_path "$MODEL_PATH" \
        --timestep 199 \
        --batch_size 3 
