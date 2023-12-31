export CUDA_VISIBLE_DEVICES='2'
python scripts/inverse.py \
    --file_id='00478.png' \
    --task_config='configs/box_inpainting_config_psld.yaml' \
    --inpainting=1 \
    --general_inverse=0 \
    --gamma=1e-1 \
    --omega=1 \
    --outdir='outputs/psld-samples-bip' 
# above gamma=1e-2 and omega=1e-1 works better for FFHQ samples
# tune for ImageNet, maybe gamma = 1e-1, omega = 1. TODO: Jun 22, 2023
