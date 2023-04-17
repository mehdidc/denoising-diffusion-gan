import math
import torch
import torchvision
import gradio as gr
from PIL import Image
import torchvision
from test_ddgan import load_model, sample
from model_configs import get_model_config


device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {
    "diffusion_db_128ch_1timesteps_openclip_vith14": load_model(get_model_config('ddgan_ddb_v2'), 'ddgan_ddb_v2.th', device=device),
    "diffusion_db_192ch_2timesteps_openclip_vith14": load_model(get_model_config('ddgan_ddb_v3'), 'ddgan_ddb_v3.th', device=device),
}
default = "diffusion_db_128ch_1timesteps_openclip_vith14"
def gen(md, model_name, md2, text, seed, nb_samples, width, height):
    torch.manual_seed(int(seed))
    model = models[model_name]
    nb_samples = int(nb_samples)
    height = int(height)
    width =  int(width)
    with torch.no_grad():
        cond = model.text_encoder([text]*nb_samples)
        if text == "":
            cond[0].normal_()
            cond[1].normal_()
            cond[0][1:] = cond[0][0:1]
            cond[1][1:] = cond[1][0:1]
            
        x_init = torch.randn(nb_samples, 3, height, width).to(device)
        fake_sample = sample(model, x_init=x_init, cond=cond)
        fake_sample = (fake_sample + 1) / 2
    grid = torchvision.utils.make_grid(fake_sample, nrow=4)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = (grid*255).astype("uint8")
    return Image.fromarray(grid)

text = """
DDGAN
"""
iface = gr.Interface(
    fn=gen,
    inputs=[
        gr.Markdown(text),
        # text caption
        gr.Dropdown(list(models.keys()), value=default), 
        gr.Markdown("If text caption is empty, random CLIP embeddings will be used as input"),
        gr.Textbox(
            lines=1, 
            placeholder="Enter text caption here, or leave empty", 
            value="Painting of a hamster king  with a crown and a cape  in a magical forest."
        ),
        gr.Number(value=0), # seed
        gr.Number(value=4), # nb_samples
        gr.Number(value=256), # width
        gr.Number(value=256),# height
    ],
    outputs="image"
)
iface.launch()
