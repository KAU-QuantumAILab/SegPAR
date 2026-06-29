import torch
from utils import seed_all
from dataset import CitySet, ADESet, VOCSet
from mmseg.apis import init_model, inference_model
import os
from PIL import Image
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt



def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def save_seg_result(config):
    # Set initial seed for reproducibility
    seed = 2
    seed_all(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    #   
    # config["num_workers"] = NUM_WORKERS
    # config["pin_memory"] = True
    # config["prefetch_factor"] = 2

    # Model configurations
    model_configs = {
        "cityscapes": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221203_045030-9a86a225.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
                "checkpoint": 'ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb2-40k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb1-80k_cityscapes-768x768.py',
                "checkpoint": 'ckpt/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth'
            }
        },
        "ade20k": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py',
                "checkpoint": 'ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
                "checkpoint": 'ckpt/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-160k_ade20k-512x512.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py',
                "checkpoint": 'ckpt/setr_pup_512x512_160k_b16_ade20k_20210619_191343-7e0ce826.pth'
            }
        },
        "VOC2012": {
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-20k_voc12aug-512x512.py',
                "checkpoint": 'ckpt/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-20k_voc12aug-512x512.py',
                "checkpoint": 'ckpt/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth'
            }
        }
    }

    # Initialize model
    if config["dataset"] not in model_configs:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    if config["model"] not in model_configs[config["dataset"]]:
        raise ValueError(f"Unsupported model: {config['model']} for dataset {config['dataset']}")
    
    model_cfg = model_configs[config["dataset"]][config["model"]]
    # model = init_model(model_cfg["config"], model_cfg["checkpoint"], device=str(device))

    if config["model"] == "setr":
        model = init_model(model_cfg["config"], None, 'cuda')
        checkpoint = torch.load(model_cfg["checkpoint"], map_location='cuda', weights_only=False)
        #  projection  bias 
        model.backbone.patch_embed.projection.bias = torch.nn.Parameter(
            torch.zeros(checkpoint["state_dict"]["backbone.patch_embed.projection.weight"].shape[0], device='cuda')
        )
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint  #   
    else:
        model = init_model(model_cfg["config"], None, 'cuda')
        # 2.   (weights_only=False  )
        checkpoint = torch.load(model_cfg["checkpoint"], map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

        del checkpoint  #   
        torch.cuda.empty_cache()  # GPU  

    # Load dataset
    if config["dataset"] == "cityscapes":
        dataset = CitySet(dataset_dir=config["data_dir"], use_gt=config["use_gt"])
    elif config["dataset"] == "ade20k":
        dataset = ADESet(dataset_dir=config["data_dir"], use_gt=config["use_gt"])
    elif config["dataset"] == "VOC2012":
        #   
        adv_dir = os.path.join(config["process_name"])
        
        # adv     
        adv_files = [os.path.join(adv_dir, f) for f in sorted(os.listdir(adv_dir))]

    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    

    # prediction
    for file in adv_files:
        result = inference_model(model, file)
        # pred_result  
        pred_dir = config["process_name"].replace("adv", "pred_result")
        os.makedirs(pred_dir, exist_ok=True)

        #      
        file_name = os.path.basename(file)
        pred_result = result.pred_sem_seg.data.squeeze().cpu().numpy()
        
        # numpy  PIL Image  PNG 
        pred_image = Image.fromarray(pred_result.astype(np.uint8))

        #     
        cmap = color_map()
        colored_result = cmap[pred_result]
        new_im = Image.fromarray(colored_result.astype(np.uint8))


        save_path = os.path.join(pred_dir, file_name.replace('.jpg', '.png'))
        new_im.save(save_path)

    


if __name__ == "__main__":
    config = {
        "dataset": "VOC2012",
        "model": "deeplabv3",
        "process_name": "results/VOC2012/deeplabv3/20250729_023143_bound_500_use_gt_False_original_mask_True_backbone_conv_use_lora_False_factor_0.05_update_valid_mask_True_lr_1e-05_l0_5.0/adv",
    }
    save_seg_result(config)