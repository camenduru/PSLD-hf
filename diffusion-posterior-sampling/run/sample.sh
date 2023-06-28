export CUDA_VISIBLE_DEVICES='1'
python3 sample_condition.py \
--model_config=configs/model_config.yaml \
--diffusion_config=configs/diffusion_config.yaml \
--task_config=configs/motion_deblur_config.yaml
# --task_config=configs/gaussian_deblur_config.yaml
# --task_config=configs/inpainting_config.yaml;
