export CUDA_VISIBLE_DEVICES='1'
python scripts/inverse.py \
    --file_id='00019.png' \
    --task_config='configs/box_inpainting_config.yaml' \
    --inpainting=1 \
    --general_inverse=0 \
    --gamma=1e-1 \
    --omega=1 \
    --W=256 \
    --H=256 \
    --scale=5.0 \
    --laion400m \
    --outdir="outputs/psld-ldm-laion400m-bip"
