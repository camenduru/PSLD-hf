wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
mkdir ./stable-diffusion/models/ldm/stable-diffusion-v1/
mv v1-5-pruned-emaonly.ckpt ./stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
