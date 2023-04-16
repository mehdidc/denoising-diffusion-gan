import os
from glob import glob
from subprocess import call
import json
def base():
    return {
        "slurm":{
            "t": 360,
            "N": 2,
            "n": 8,
        },
        "model":{
            "dataset": "wds",
            "seed": 0,
            "cross_attention": False,
            "num_channels": 3,
            "centered": True,
            "use_geometric": False,
            "beta_min": 0.1,
            "beta_max": 20.0,
            "num_channels_dae": 128,
            "n_mlp": 3,
            "ch_mult": [1, 1, 2, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.0,
            "resamp_with_conv": True,
            "conditional": True,
            "fir": True,
            "fir_kernel": [1, 3, 3, 1],
            "skip_rescale": True,
            "resblock_type": "biggan",
            "progressive": "none",
            "progressive_input": "residual",
            "progressive_combine": "sum",
            "embedding_type": "positional",
            "fourier_scale": 16.0,
            "not_use_tanh": False,
            "image_size": 256,
            "nz": 100,
            "num_timesteps": 4,
            "z_emb_dim": 256,
            "t_emb_dim": 256,
            "text_encoder": "google/t5-v1_1-base",
            "masked_mean": True,
            "cross_attention_block": "basic",
        }
    }
def ddgan_cc12m_v2():
    cfg =  base()
    cfg['slurm']['N'] = 2
    cfg['slurm']['n'] = 8
    return cfg

def ddgan_cc12m_v6():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"
    return cfg

def ddgan_cc12m_v7():
    cfg = base()
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['slurm']['N'] = 2
    cfg['slurm']['n'] = 8
    return cfg

def ddgan_cc12m_v8():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    return cfg

def ddgan_cc12m_v9():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['model']['num_channels_dae'] = 320
    cfg['model']['image_size'] = 64
    cfg['model']['batch_size'] = 1
    return cfg

def ddgan_cc12m_v11():
    cfg = base()
    cfg['model']['text_encoder'] = "google/t5-v1_1-large"    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['model']['cross_attention'] = True
    return cfg

def ddgan_cc12m_v12():
    cfg = ddgan_cc12m_v11()
    cfg['model']['text_encoder'] = "google/t5-v1_1-xl"    
    cfg['model']['preprocessing'] = 'random_resized_crop_v1'
    return cfg

def ddgan_cc12m_v13():
    cfg = ddgan_cc12m_v12()
    cfg['model']['discr_type'] = "large_cond_attn"
    return cfg

def ddgan_cc12m_v14():
    cfg = ddgan_cc12m_v12()
    cfg['model']['num_channels_dae'] = 192
    return cfg

def ddgan_cc12m_v15():
    cfg = ddgan_cc12m_v11()
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg

def ddgan_cifar10_cond17():
    cfg = base()
    cfg['model']['image_size'] = 32    
    cfg['model']['classifier_free_guidance_proba'] = 0.2
    cfg['model']['ch_mult'] = "1 2 2 2"
    cfg['model']['cross_attention'] = True
    cfg['model']['dataset'] = "cifar10"
    cfg['model']['n_mlp'] = 4
    return cfg

def ddgan_cifar10_cond18():
    cfg = ddgan_cifar10_cond17()
    cfg['model']['text_encoder'] = "google/t5-v1_1-xl"    
    return cfg

def ddgan_cifar10_cond19():
    cfg = ddgan_cifar10_cond17()
    cfg['model']['discr_type'] = 'small_cond_attn'
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg

def ddgan_laion_aesthetic_v1():
    cfg = ddgan_cc12m_v11()
    cfg['model']['dataset_root'] = '"/p/scratch/ccstdl/cherti1/LAION-aesthetic/output/{00000..05038}.tar"'
    return cfg

def ddgan_laion_aesthetic_v2():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['discr_type'] = "large_cond_attn"
    return cfg

def ddgan_laion_aesthetic_v3():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['text_encoder'] = "google/t5-v1_1-xl" 
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg

def ddgan_laion_aesthetic_v4():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['text_encoder'] = "openclip/ViT-L-14-336/openai" 
    return cfg


def ddgan_laion_aesthetic_v5():
    cfg = ddgan_laion_aesthetic_v1()
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    return cfg



def ddgan_laion2b_v1():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['mismatch_loss'] = ''
    cfg['model']['grad_penalty_cond'] = ''
    cfg['model']['num_channels_dae'] = 224
    cfg['model']['batch_size'] = 2
    cfg['model']['discr_type'] = "large_cond_attn"
    cfg['model']['preprocessing'] = 'random_resized_crop_v1'
    return cfg

def ddgan_laion_aesthetic_v6():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['no_lr_decay'] = ''
    return cfg



def ddgan_laion_aesthetic_v7():
    cfg = ddgan_laion_aesthetic_v6()
    cfg['model']['r1_gamma'] = 5
    return cfg


def ddgan_laion_aesthetic_v8():
    cfg = ddgan_laion_aesthetic_v6()
    cfg['model']['num_timesteps'] = 8
    return cfg

def ddgan_laion_aesthetic_v9():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['num_channels_dae'] = 384
    return cfg

def ddgan_sd_v1():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v2():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v3():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v4():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v5():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['num_timesteps'] = 8
    return cfg
def ddgan_sd_v6():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['num_channels_dae'] = 192
    return cfg
def ddgan_sd_v7():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_sd_v8():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['image_size'] = 512
    return cfg
def ddgan_laion_aesthetic_v12():
    cfg = ddgan_laion_aesthetic_v3()
    return cfg
def ddgan_laion_aesthetic_v13():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-H-14/laion2b_s32b_b79k" 
    return cfg

def ddgan_laion_aesthetic_v14():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-H-14/laion2b_s32b_b79k" 
    return cfg
def ddgan_sd_v9():
    cfg = ddgan_laion_aesthetic_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-H-14/laion2b_s32b_b79k" 
    return cfg

def ddgan_sd_v10():
    cfg = ddgan_sd_v9()
    cfg['model']['num_timesteps'] = 2
    return cfg

def ddgan_laion2b_v2():
    cfg = ddgan_sd_v9()
    return cfg

def ddgan_ddb_v1():
    cfg = ddgan_sd_v10()
    return cfg

def ddgan_sd_v11():
    cfg = ddgan_sd_v10()
    cfg['model']['image_size'] = 512
    return cfg

def ddgan_ddb_v2():
    cfg = ddgan_ddb_v1()
    cfg['model']['num_timesteps'] = 1
    return cfg

def ddgan_ddb_v3():
    cfg = ddgan_ddb_v1()
    cfg['model']['num_channels_dae'] = 192
    cfg['model']['num_timesteps'] = 2
    return cfg

def ddgan_ddb_v4():
    cfg = ddgan_ddb_v1()
    cfg['model']['num_channels_dae'] = 256
    cfg['model']['num_timesteps'] = 2
    return cfg

def ddgan_ddb_v5():
    cfg = ddgan_ddb_v2()
    return cfg

def ddgan_ddb_v6():
    cfg = ddgan_ddb_v3()
    return cfg

def ddgan_ddb_v7():
    cfg = ddgan_ddb_v1()
    return cfg

def ddgan_ddb_v9():
    cfg = ddgan_ddb_v3()
    cfg['model']['attn_resolutions'] = [4, 8, 16, 32]
    return cfg

def ddgan_laion_aesthetic_v15():
    cfg = ddgan_ddb_v3()
    return cfg

def ddgan_ddb_v10():
    cfg = ddgan_ddb_v9()
    return cfg

def ddgan_ddb_v11():
    cfg = ddgan_ddb_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-g-14/laion2B-s12B-b42K" 
    return cfg

def ddgan_ddb_v12():
    cfg = ddgan_ddb_v3()
    cfg['model']['text_encoder'] = "openclip/ViT-bigG-14/laion2b_s39b_b160k"
    return cfg


models = [
    ddgan_cifar10_cond17, # cifar10, cross attn for discr
    ddgan_cifar10_cond18, # cifar10, xl encoder
    ddgan_cifar10_cond19, # cifar10, xl encoder

    ddgan_cc12m_v2, # baseline (no large text encoder, no classifier guidance)
    ddgan_cc12m_v6, # like v2 but using large T5 text encoder
    ddgan_cc12m_v7, # like v2 but with classifier guidance
    ddgan_cc12m_v8, # like v6 but classifier guidance
    ddgan_cc12m_v9, # ~1B model but 64x64 resolution
    ddgan_cc12m_v11, # large text encoder + cross attention + classifier free guidance
    ddgan_cc12m_v12, # T5-XL + cross attention + classifier free guidance + random_resized_crop_v1
    ddgan_cc12m_v13, # T5-XL + cross attention + classifier free guidance + random_resized_crop_v1 + cond attn
    ddgan_cc12m_v14, # T5-XL + cross attention + classifier free guidance + random_resized_crop_v1 + 300M model
    ddgan_cc12m_v15, # fine-tune v11 with --mismatch_loss and --grad_penalty_cond

    ddgan_laion_aesthetic_v1, # like ddgan_cc12m_v11 but fine-tuned on laion aesthetic
    ddgan_laion_aesthetic_v2, # like ddgan_laion_aesthetic_v1 but trained from scratch with the new cross attn discr
    ddgan_laion_aesthetic_v3, # like ddgan_laion_aesthetic_v1 but trained from scratch with T5-XL (continue from 23aug with mismatch and grad penalty and random_resized_crop_v1)
    ddgan_laion_aesthetic_v4, # like ddgan_laion_aesthetic_v1 but trained from scratch with OpenAI's ClipEncoder 
    ddgan_laion_aesthetic_v5, # fine-tune ddgan_laion_aesthetic_v1 with mismatch and cond grad penalty  losses
    ddgan_laion_aesthetic_v6, # like v3 but without lr decay
    ddgan_laion_aesthetic_v7, # like v6 but  with r1 gamma of 5 instead of 1, trying to constrain the discr more.
    ddgan_laion_aesthetic_v8, # like v6 but with 8 timesteps
    ddgan_laion_aesthetic_v9,
    ddgan_laion_aesthetic_v12,
    ddgan_laion_aesthetic_v13,
    ddgan_laion_aesthetic_v14,
    ddgan_laion_aesthetic_v15,
    ddgan_laion2b_v1,
    ddgan_sd_v1,
    ddgan_sd_v2,
    ddgan_sd_v3,
    ddgan_sd_v4,
    ddgan_sd_v5,
    ddgan_sd_v6,
    ddgan_sd_v7,
    ddgan_sd_v8,
    ddgan_sd_v9,
    ddgan_sd_v10,
    ddgan_sd_v11,
    ddgan_laion2b_v2,
    ddgan_ddb_v1,
    ddgan_ddb_v2,
    ddgan_ddb_v3,
    ddgan_ddb_v4,
    ddgan_ddb_v5,
    ddgan_ddb_v6,
    ddgan_ddb_v7,
    ddgan_ddb_v9,
    ddgan_ddb_v10,
    ddgan_ddb_v11,
    ddgan_ddb_v12,
]

def get_model_config(model_name):
    for model in models:
        if model.__name__ == model_name:
            return model()['model']