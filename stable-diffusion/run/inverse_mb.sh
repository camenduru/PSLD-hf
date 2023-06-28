export CUDA_VISIBLE_DEVICES='0'
python scripts/inverse.py \
    --file_id='00019.png' \
    --task_config='configs/motion_deblur_config_psld.yaml' \
    --outdir='outputs/psld-samples-mb';