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

### (3) ID file 지정

# ID_FILE="/local_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/test_ids_prior_model.txt"
ID_FILE="/local_data_2/urp25su_sbpark/2dgs_prior_rendered/dataset/test_ids.txt"


### (4) Inference하고자 하는 IDs가 있는 상위 폴더의 경로

# ROOT="/local_data_2/dataset/difix3d_face_dataset_video/20250729_unetattn_nblk1_Eexpr_dexpr128_id334_bs32_nview4_lr0.0002_4viewagg_rgbnrmmask_LgeomuvLnrmuv_noTV_epoch10000"
ROOT="/local_data_2/dataset/difix3d_face_dataset_video/20250729_unetattn_nblk1_Eexpr_dexpr128_id334_bs32_nview4_lr0.0002_4viewagg_rgbnrmmask_LgeomuvLnrmuv_noTV_personalize_ep10000"
# ROOT="/local_data_2/dataset/difix3d_face_dataset/20250707_unetattn_nblk1_dexpr128_id334_bs32_nview4_lr0.0002_4viewagg_resizepad_rgbnrmmask_epoch10000"

# ROOT="/local_data_2/dataset/difix3d_face_dataset_video/20250729_unetattn_nblk1_Eexpr_dexpr128_id334_bs32_nview4_lr0.0002_4viewagg_rgbnrmmask_LgeomuvLnrmuv_noTV_personalize_ep10000_multiexpr"


### (5) Output 저장할 path와 model의 ckpt 지정
OUT_ROOT="/local_data_2/urp25su_sbpark/Difix3D/outputs_multi/videos/random_mask/"

MODEL_PATH="/local_data_2/urp25su_sbpark/Difix3D/outputs/difix/train/checkpoints_random_mask/model.pkl"

### (6) ID file에 지정된 IDs의 이름을 갖는 폴더 아래의 이미지들을 inference
# 간략한 옵션 설명
# --video: inference 결과를 video 형태로 저장 
# --frame_rate: video의 frame rate를 지정
# --batch_size: batch 단위 inference를 위한 옵션
# --record_time: inference 시에 소요된 시간을 측정해서 파일 형태로 저장한다.

while IFS= read -r ID; do
    echo "=== Running inference for ID: $ID ==="

    python src/inference_difix_video_copy.py \
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
        

    echo ""
done < "$ID_FILE"

# python src/inference_difix_video.py \
#     --input_image "$ROOT" \
#     --ref_image "$ROOT_REF" \
#     --output_dir "$OUT_ROOT/150" \
#     --model_path "$MODEL_PATH" \
#     --prompt "remove degradation" \
#     --timestep 199 \
#     --video \
#     --frame_rate 30 \
#     --batch_size 2 \
#     --height 400 \
#     --width 272