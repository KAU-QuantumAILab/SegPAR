import torch
import numpy as np
from mmengine.config import Config
from mmseg.models import build_segmentor
from mmengine.runner import load_checkpoint
from mmseg.datasets import CityscapesDataset
from torch.utils.data import DataLoader
from mmcv.transforms import Compose
from tqdm import tqdm
from mmengine.registry import init_default_scope
from mmengine.dataset import default_collate
from mmseg.evaluation import IoUMetric
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

from mmseg.apis import init_model, inference_model
from dataset import VOCSet
import evaluate
import os
from PIL import Image




# 4.  
test_dataset = VOCSet('datasets/VOC2012')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


CITYSCAPES_LABEL_MAPPING = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18   # bicycle
}

def convert_gt_labels(gt_array):
    """
    GT  (gtFine_labelIds) Cityscapes   trainId .
       ignore index 255 .
    """
    ignore_val = 255
    converted = np.full_like(gt_array, ignore_val)
    for orig_label, train_label in CITYSCAPES_LABEL_MAPPING.items():
        converted[gt_array == orig_label] = train_label
    return converted


def cal_miou(config):
    # 1. Config & Model 
    model = init_model(config["cf_path"], config["ckpt_path"], 'cuda')
    mean_iou = evaluate.load("mean_iou", "segmentation")

    #   
    adv_dir = os.path.join(config["process_name"])
    gt_dir = os.path.join(config["gt_dir"])
    
    # adv     
    adv_files = [os.path.join(adv_dir, f) for f in sorted(os.listdir(adv_dir))]
    gt_files = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))]

    pred_list = []
    gt_list = []

    with torch.no_grad():
        for file, gt_file in zip(adv_files, gt_files):
            img_np = np.array(Image.open(file))[:,:,::-1]
            gt_np = np.array(Image.open(gt_file))




            result = inference_model(model, img_np)
            pred_list.append(result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8))
            gt_list.append(gt_np)


        iou = mean_iou.compute(
            predictions=pred_list,
            references=gt_list,
            num_labels=21,
            ignore_index=255,
            reduce_labels=False,
        )
        print(iou)
        print(iou['per_category_iou'][1:], sum(iou['per_category_iou'][1:])/20)



if __name__ == "__main__":
    config = {
        "dataset": "VOC2012",
        "model": "deeplabv3",
        "gt_dir": "./datasets/VOC2012/SegmentationClass",
        "process_name": "results/VOC2012/deeplabv3/20250729_023143_bound_500_use_gt_False_original_mask_True_backbone_conv_use_lora_False_factor_0.05_update_valid_mask_True_lr_1e-05_l0_5.0/adv",
        "cf_path": "./configs/deeplabv3/deeplabv3_r101-d8_4xb4-20k_voc12aug-512x512.py",
        "ckpt_path": "./ckpt/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth",
    }
    cal_miou(config)

