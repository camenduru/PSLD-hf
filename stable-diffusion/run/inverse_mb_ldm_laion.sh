export CUDA_VISIBLE_DEVICES='1'
python scripts/inverse.py \
    --file_id='00019.png' \
    --task_config='configs/motion_deblur_config.yaml' \
    --inpainting=0 \
    --general_inverse=1 \
    --gamma=1e-1 \
    --omega=1 \
    --W=256 \
    --H=256 \
    --scale=5.0 \
    --laion400m \
    --outdir="outputs/psld-ldm-laion400m-mb"
