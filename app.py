import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from share_btn import community_icon_html, loading_icon_html
from tqdm.auto import tqdm
from PIL import Image


# PARAMS
MANUAL_SEED = 42
HEIGHT = 512
WIDTH = 512
ETA = 1e-1


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)  
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(torch_device)

# 1. Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
scheduler = LMSDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device) 


generator = torch.manual_seed(MANUAL_SEED)   # Seed generator to create the inital latent noise


def read_content(file_path: str) -> str:
    """read the content of target file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content

# def predict(dict, prompt=""):
#     init_image = dict["image"].convert("RGB").resize((512, 512))
#     mask = dict["mask"].convert("RGB").resize((512, 512))
#     output = pipe(prompt = prompt, image=init_image, mask_image=mask,guidance_scale=7.5)
#     return output.images[0], gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def predict(dict, prompt=""):
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]   

    init_image = dict["image"].convert("RGB").resize((512, 512))
    mask = dict["mask"].convert("RGB").resize((512, 512))

    # convert input image to array in [-1, 1]
    init_image = torch.tensor(2 * (np.asarray(init_image) / 255) - 1, device=torch_device)
    mask = torch.tensor((np.asarray(mask) / 255), device=torch_device)
    # add one dimension for the batch and bring channels first
    init_image = init_image.permute(2, 0, 1).unsqueeze(0)
    mask = mask.permute(2, 0, 1).unsqueeze(0)

    latents = torch.randn(
    (1, unet.in_channels, HEIGHT // 8, WIDTH // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)

    for i, t in enumerate(tqdm(scheduler.timesteps)):
        t = scheduler.timesteps[i]            
        z_t = torch.clone(latents.detach())
        z_t.requires_grad = True

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = scheduler.scale_model_input(z_t, t)
        
        
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=uncond_embeddings).sample
        # compute z_0 using tweedies's formula
        indx = scheduler.num_inference_steps - i - 1
        z_0 = (1/torch.sqrt(scheduler.alphas_cumprod[indx]))\
                    *(z_t + (1-scheduler.alphas_cumprod[indx]) * noise_pred )

        # pass through the decoder
        z_0 = 1 / 0.18215 * z_0
        image_pred = vae.decode(z_0).sample
        # clip
        image_pred = torch.clamp(image_pred, min=-1.0, max=1.0)
        inpainted_image = (1 - mask) * init_image + mask * image_pred    
        error_measurement = (1/2) * torch.linalg.norm((1-mask) * (init_image - image_pred))**2
        # TODO(giannisdaras): add LPIPS?
        error = error_measurement
        gradients = torch.autograd.grad(error, inputs=z_t)[0]
        # compute the previous noisy sample x_t -> x_t-1
        z_t_next = scheduler.step(noise_pred, t, z_t).prev_sample
        
        latents = z_t_next - ETA * gradients
    
    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    return images[0], gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


css = '''
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
'''

image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    gr.HTML(read_content("header.html"))
    with gr.Group():
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload").style(height=400)
                    with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                        prompt = gr.Textbox(placeholder = 'Your prompt (what you want in place of what is erased)', show_label=False, elem_id="input-text")
                        btn = gr.Button("Inpaint!").style(
                            margin=False,
                            rounded=(False, True, True, False),
                            full_width=False,
                        )
                with gr.Column():
                    image_out = gr.Image(label="Output", elem_id="output-img").style(height=400)
                    with gr.Group(elem_id="share-btn-container"):
                        community_icon = gr.HTML(community_icon_html, visible=False)
                        loading_icon = gr.HTML(loading_icon_html, visible=False)            

            btn.click(fn=predict, inputs=[image, prompt], outputs=[image_out, community_icon, loading_icon])



image_blocks.launch()