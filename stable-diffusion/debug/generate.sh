export CUDA_VISIBLE_DEVICES='2'
python scripts/inverse.py \
    --file_id='00019.png' \
    --task_config='configs/motion_deblur_config.yaml' \
    --inpainting=0 \
    --general_inverse=0 \
    --gamma=1e-1 \
    --omega=1e-1 \
    --W=256 \
    --H=256 \
    --scale=5.0 \
    --laion400m \
    --prompt="a photograph of fantasy landscape trending in art station" \
    --outdir="outputs/txt2img-samples-laion400m"