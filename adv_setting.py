import torch
import numpy as np
import cv2
import os
import sys
import evaluate
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
from dataset import CitySet
# ddcat  import
from adv_models.pspnet import PSPNet, DeepLabV3, PSPNet_DDCAT, DeepLabV3_DDCAT
#   Python path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#   dataset  import
from dataset import CitySet, ADESet, VOCSet
cv2.ocl.setUseOpenCL(False)
import torch, torch.nn.functional as F, numpy as np, cv2, math
from contextlib import nullcontext
#     
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def load_model(config):
        """    """
        if config["model"] == 'deeplabv3_sat':
            return DeepLabV3(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        elif config["model"] == 'deeplabv3_ddcat':
            return DeepLabV3_DDCAT(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        elif config["model"] == 'pspnet_sat':
            return PSPNet(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        elif config["model"] == 'pspnet_ddcat':
            return PSPNet_DDCAT(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        else:
            raise ValueError(f"   : {config['model']}")
        
    








# ──────────   ──────────
torch.backends.cudnn.benchmark = True      #     


@torch.no_grad()
def _get_norm_tensors(config, mean, std):
    device = torch.device(config["device"])
    cache = config.get("_norm_cache")
    cache_key = (str(device), tuple(mean), None if std is None else tuple(std))
    if cache is None or cache.get("key") != cache_key:
        mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        std_t = None if std is None else torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        config["_norm_cache"] = {"key": cache_key, "mean": mean_t, "std": std_t}
    return config["_norm_cache"]["mean"], config["_norm_cache"]["std"]


@torch.no_grad()
def fast_net_process(model, batch_imgs_np, mean, std=None, do_flip=True, amp=True, config=None):
    """
     crop    (B,H,W,C)  .
    """
    device = torch.device(config["device"])
    batch = torch.from_numpy(np.stack(batch_imgs_np, axis=0)).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    mean_t, std_t = _get_norm_tensors(config, mean, std)

    if std_t is None:
        batch = batch - mean_t
    else:
        batch = (batch - mean_t) / std_t

    model_in = torch.cat([batch, batch.flip(3)], dim=0) if do_flip else batch
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=True)
        if amp and device.type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        out = model(model_in)

    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.shape[-2:] != batch.shape[-2:]:
        out = F.interpolate(out, batch.shape[-2:], mode="bilinear", align_corners=False)

    out = F.softmax(out, dim=1)
    if do_flip:
        b = batch.shape[0]
        out = (out[:b] + out[b:].flip(3)) * 0.5

    return out.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)

def fast_scale_process(model, image, classes,
                       crop_h, crop_w, h, w, mean, std=None,
                       stride_rate=2/3, batch_size=8, amp=True, config=None):
    device = torch.device(config["device"])

    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    ph1, pw1 = pad_h // 2, pad_w // 2
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, ph1, pad_h - ph1, pw1, pad_w - pw1,
                                   cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w = image.shape[:2]
    sh, sw = math.ceil(crop_h * stride_rate), math.ceil(crop_w * stride_rate)
    gh, gw = math.ceil((new_h - crop_h) / sh) + 1, math.ceil((new_w - crop_w) / sw) + 1

    # ─────  crop   
    coords = []
    for ih in range(gh):
        for iw in range(gw):
            s_h = max(min(ih * sh, new_h - crop_h), 0)
            s_w = max(min(iw * sw, new_w - crop_w), 0)
            coords.append((s_h, s_w, s_h + crop_h, s_w + crop_w))

    pred_map = torch.zeros((new_h, new_w, classes), device=device, dtype=torch.float32)
    cnt_map = torch.zeros((new_h, new_w, 1), device=device, dtype=torch.float32)

    # ─────   
    model.eval()
    for i in range(0, len(coords), batch_size):
        batch_imgs = [image[y1:y2, x1:x2] for (y1,x1,y2,x2) in coords[i:i+batch_size]]
        batch_outs = fast_net_process(model, batch_imgs, mean, std, amp=amp, config=config)

        for j, (y1, x1, y2, x2) in enumerate(coords[i:i + batch_size]):
            pred_map[y1:y2, x1:x2] += batch_outs[j]
            cnt_map[y1:y2, x1:x2] += 1.0

    pred_map = pred_map / torch.clamp(cnt_map, min=1.0)
    pred_map = pred_map[ph1:ph1 + ori_h, pw1:pw1 + ori_w]
    pred_map = pred_map.permute(2, 0, 1).unsqueeze(0)
    pred_map = F.interpolate(pred_map, size=(h, w), mode="bilinear", align_corners=False)
    pred_map = pred_map.squeeze(0).permute(1, 2, 0).contiguous()
    return pred_map


def model_predict(model, image, config):
    device = torch.device(config["device"])
    image = image[:,:,::-1]             # bgr -> rgb
    h, w, _ = image.shape

    if config["dataset"] == "cityscapes":
        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)

    
    
    confidence = torch.zeros((h, w, config["num_class"]), device=device, dtype=torch.float32)
    for scale in config["scales"]:
        long_size = round(scale * config["base_size"])
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size/float(h)*w)
        else:
            new_h = round(long_size/float(w)*h)

        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        confidence += fast_scale_process(
            model,
            image_scale,
            config["num_class"],
            config["crop_h"],
            config["crop_w"],
            h,
            w,
            config["mean"],
            config["std"],
            config=config,
        )

    confidence = confidence / len(config["scales"])
    confidence = confidence.permute(2, 0, 1)
    prediction = torch.argmax(confidence, dim=0)
    return confidence, prediction

def main(config):
    #  
    set_seed(42)
    
    model = load_model(config)
    checkpoint = torch.load(config["model_path"])
    #  cuda   DataParallel 
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    
    
    model.eval()
    
    # CUDA   (  )
    cudnn.benchmark = False


    mean_iou = evaluate.load("mean_iou", "segmentation")

    if config["dataset"] == "cityscapes":
        dataset = CitySet("datasets/cityscapes", use_gt=False)
        
    elif config["dataset"] == "VOC2012":
        dataset = VOCSet(config)
        adv_files = dataset.adv_files

    # Normalization parameters
    value_scale = 255

    mean_rgb = [0.485, 0.456, 0.406]  # [R, G, B]
    mean_rgb = [item * value_scale for item in mean_rgb]
    std_rgb = [0.229, 0.224, 0.225]   # [R, G, B]
    std_rgb = [item * value_scale for item in std_rgb]
    


    pred_list = []
    gt_list = []

    for i in range(len(dataset)):
        image, _, gt = dataset[i]
        confidence, prediction = model_predict(model, image, config)
        # confidence (H,W) according to prediction
        # prediction (H,W)
        pred_list.append(prediction)
        gt_list.append(gt)
    
    iou = mean_iou.compute(
        predictions=pred_list,
        references=gt_list,
        num_labels=config["num_class"],
        ignore_index=255,
    )
    print(f"Mean IoU: {iou}")




if __name__ == "__main__":
    config = {
        "model": "pspnet_sat",
        "model_path": "adv_models/pretrain/cityscapes/pspnet/sat/train_epoch_400.pth",
        "device": "cuda",
        "dataset": "cityscapes",
        "layers": 50,
        "num_class": 19,
        "zoom_factor": 8,
        "base_size": 1024,
        "crop_h": 449,
        "crop_w": 449,
        "scales": [1.0],
        "mean":[255*0.485, 255*0.456, 255*0.406],   # [R, G, B]
        "std":[255*0.229, 255*0.224, 255*0.225]     # [R, G, B]
    }
    main(config)
