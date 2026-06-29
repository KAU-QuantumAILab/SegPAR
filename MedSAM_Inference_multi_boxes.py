# -*- coding: utf-8 -*-
"""
   bounding box   ,   forward   
image-level prediction class map(= ID)  .

usage example:
  # PNG +  bbox + class id
  python MedSAM_Inference_multi_boxes.py -i assets/img_demo.png -o ./ --boxes "[95,255,190,350]" --class_ids 1
  python MedSAM_Inference_multi_boxes.py -i assets/img_demo.png -o ./ --boxes "[95,255,190,350]" "[200,100,300,280]" --class_ids 1 3

  # data/npy  (GT bbox/label  )
  python MedSAM_Inference_multi_boxes.py --npy_root data/npy/CT_Abd -o ./ --npy_index 0
  python MedSAM_Inference_multi_boxes.py --npy_root data/npy/CT_Abd -o ./ --npy_name CT_Abd_FLARE22_Tr_0001-000.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse

join = os.path.join


def load_medsam_model(
    checkpoint: str = "work_dir/MedSAM/medsam_vit_b.pth",
    device: str = "cuda:0",
):
    """
        MedSAM  .
    """

    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model = model.to(device)
    model.eval()
    return model


def _prepare_image_to_1024(img_np: np.ndarray, device: str):
    """
       MedSAM  (1024  ) .
    """
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np

    H, W = img_3c.shape[0], img_3c.shape[1]
    # npy (1024x1024, float [0,1])    
    # MedSAM-main npy main    .
    if (
        H == 1024
        and W == 1024
        and np.issubdtype(img_3c.dtype, np.floating)
        and float(np.min(img_3c)) >= 0.0
        and float(np.max(img_3c)) <= 1.0
    ):
        img_1024_tensor = torch.tensor(img_3c).float().permute(2, 0, 1).unsqueeze(0).to(device)
        return img_3c, H, W, img_1024_tensor

    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return img_3c, H, W, img_1024_tensor


def _parse_boxes_arg(boxes_arg: list[str]) -> np.ndarray:
    """
    CLI --boxes   (N,4) int64  .
    """
    box_list = []
    for s in boxes_arg:
        s = s.strip().strip("[]")
        coords = [int(x.strip()) for x in s.split(",")]
        if len(coords) != 4:
            raise ValueError(f"bbox 4  : {s}")
        box_list.append(coords)
    if len(box_list) == 0:
        raise ValueError(" 1  bbox .")
    return np.array(box_list, dtype=np.int64)


def load_dataset_sample_from_image(
    data_path: str,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
) -> dict:
    """
    [  ]   + bbox/class_id   sample dict .
    """
    img_np = io.imread(data_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError(f"boxes_xyxy (N,4) .  shape={boxes_xyxy.shape}")
    if class_ids.ndim != 1 or len(class_ids) != len(boxes_xyxy):
        raise ValueError(
            f"class_ids ({len(class_ids)}) bbox ({len(boxes_xyxy)})  ."
        )
    if np.any(class_ids <= 0):
        raise ValueError("class_ids 0   . (0  reserved)")

    return {
        "image_rgb": img_3c,
        "boxes_xyxy": boxes_xyxy,
        "class_ids": class_ids,
        "gt": None,
        "base_name": os.path.splitext(os.path.basename(data_path))[0],
        "source": "image",
    }


def load_dataset_sample_from_npy_root(
    npy_root: str,
    npy_index: int = 0,
    npy_name: str | None = None,
    bbox_shift: int = 20,
) -> dict:
    """
    [  ] data/npy (imgs, gts) 1    sample dict .
    """
    imgs_dir = join(npy_root, "imgs")
    gts_dir = join(npy_root, "gts")
    gt_files = sorted(glob.glob(join(gts_dir, "**/*.npy"), recursive=True))
    gt_files = [f for f in gt_files if os.path.isfile(join(imgs_dir, os.path.basename(f)))]
    if not gt_files:
        raise FileNotFoundError(f"   : {npy_root}/imgs, gts")

    if npy_name:
        gt_path = join(gts_dir, npy_name)
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(gt_path)
        img_name = npy_name
    else:
        idx = min(npy_index, len(gt_files) - 1)
        gt_path = gt_files[idx]
        img_name = os.path.basename(gt_path)
    img_path = join(imgs_dir, img_name)

    img_1024 = np.load(img_path, allow_pickle=True)
    gt = np.load(gt_path, allow_pickle=True)
    H, W = img_1024.shape[0], img_1024.shape[1]
    if not (H == 1024 and W == 1024):
        raise ValueError("npy  1024x1024  .")

    boxes_1024, class_ids = get_bboxes_and_labels_from_gt(gt, bbox_shift=bbox_shift)
    if boxes_1024.shape[0] == 0:
        raise ValueError("GT  bbox  (  ).")

    return {
        "image_rgb": img_1024,
        "boxes_xyxy": boxes_1024.astype(np.int64),  # npy   1024
        "class_ids": class_ids,
        "gt": gt,
        "base_name": os.path.splitext(img_name)[0],
        "source": "npy",
    }


def _collect_npy_pairs(npy_root: str) -> list[tuple[str, str, str]]:
    """
    npy_root/imgs, npy_root/gts  .
    Returns:
        [(img_path, gt_path, img_name), ...]
    """
    imgs_dir = join(npy_root, "imgs")
    gts_dir = join(npy_root, "gts")
    gt_files = sorted(glob.glob(join(gts_dir, "**/*.npy"), recursive=True))
    pairs = []
    for gt_path in gt_files:
        img_name = os.path.basename(gt_path)
        img_path = join(imgs_dir, img_name)
        if os.path.isfile(img_path):
            pairs.append((img_path, gt_path, img_name))
    if not pairs:
        raise FileNotFoundError(f"   : {npy_root}/imgs, gts")
    return pairs


def load_dataset_from_npy_root(
    npy_root: str,
    bbox_shift: int = 20,
    max_items: int | None = None,
    skip_empty_gt: bool = True,
) -> list[dict]:
    """
    [  ] npy_root  sample dict  .
    """
    pairs = _collect_npy_pairs(npy_root)
    if max_items is not None:
        pairs = pairs[: max_items]

    dataset = []
    for img_path, gt_path, img_name in pairs:
        img_1024 = np.load(img_path, allow_pickle=True)
        gt = np.load(gt_path, allow_pickle=True)
        H, W = img_1024.shape[0], img_1024.shape[1]
        if not (H == 1024 and W == 1024):
            raise ValueError(f"npy  1024x1024 : {img_name}, shape={img_1024.shape}")

        boxes_1024, class_ids = get_bboxes_and_labels_from_gt(gt, bbox_shift=bbox_shift)
        if boxes_1024.shape[0] == 0:
            if skip_empty_gt:
                continue
            raise ValueError(f"GT  bbox : {img_name}")

        dataset.append(
            {
                "image_rgb": img_1024,
                "boxes_xyxy": boxes_1024.astype(np.int64),
                "class_ids": class_ids,
                "gt": gt,
                "base_name": os.path.splitext(img_name)[0],
                "source": "npy",
            }
        )
    return dataset


def get_bboxes_and_labels_from_gt(gt: np.ndarray, bbox_shift: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    GT   bbox label id  (1024 ).
    """
    label_ids = np.unique(gt)
    label_ids = label_ids[label_ids > 0]
    boxes = []
    labels = []
    H, W = gt.shape
    for lb in label_ids:
        y_indices, x_indices = np.where(gt == lb)
        if len(x_indices) == 0:
            continue
        x_min = max(0, np.min(x_indices) - bbox_shift)
        x_max = min(W, np.max(x_indices) + bbox_shift)
        y_min = max(0, np.min(y_indices) - bbox_shift)
        y_max = min(H, np.max(y_indices) + bbox_shift)
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(int(lb))

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def get_bboxes_from_gt(gt: np.ndarray, bbox_shift: int = 20):
    """
     : bbox    .
    """
    boxes, _ = get_bboxes_and_labels_from_gt(gt, bbox_shift=bbox_shift)
    return boxes


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, color="blue"):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def medsam_inference_multi_boxes_with_probs(
    medsam_model,
    img_embed,
    boxes_1024,
    H,
    W,
    threshold: float = 0.5,
):
    """
        bbox   /   .

    Args:
        medsam_model: MedSAM 
        img_embed: (1, 256, 64, 64)  
        boxes_1024: (N, 4) 1024  bbox [x1,y1,x2,y2]
        H, W:   , 
        threshold:   

    Returns:
        prob_maps: (N, H, W) float32 
        masks: (N, H, W) uint8  
    """
    N = boxes_1024.shape[0]
    device = img_embed.device
    box_torch = torch.as_tensor(boxes_1024, dtype=torch.float, device=device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (N, 1, 4)

    #  1  N  bbox   
    img_embed_batch = img_embed.repeat(N, 1, 1, 1)  # (N, 256, 64, 64)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed_batch,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    # (N, 1, 256, 256) -> (N, 1, H, W)
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )
    prob_maps = low_res_pred.squeeze(1).cpu().numpy().astype(np.float32)
    masks = (prob_maps > threshold).astype(np.uint8)
    return prob_maps, masks  # (N, H, W), (N, H, W)


@torch.no_grad()
def medsam_inference_multi_boxes(medsam_model, img_embed, boxes_1024, H, W):
    """
     :    (  ).
    """
    _, masks = medsam_inference_multi_boxes_with_probs(
        medsam_model=medsam_model,
        img_embed=img_embed,
        boxes_1024=boxes_1024,
        H=H,
        W=W,
    )
    return masks


def build_prediction_class_map(
    prob_maps: np.ndarray,
    class_ids: np.ndarray,
    threshold: float = 0.5,
    background_id: int = 0,
) -> np.ndarray:
    """
     bbox   image-level class map .
    -    bbox class_id 
    -   threshold  background_id 
    """
    if prob_maps.ndim != 3:
        raise ValueError(f"prob_maps (N,H,W) .  shape={prob_maps.shape}")
    n_masks, H, W = prob_maps.shape
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if class_ids.ndim != 1 or len(class_ids) != n_masks:
        raise ValueError(
            f"class_ids ({len(class_ids)}) bbox/mask ({n_masks})  ."
        )
    if np.any(class_ids <= 0):
        raise ValueError("class_ids 0   . (0 )")
    if background_id != 0:
        raise ValueError("  background_id=0 .")

    best_idx = np.argmax(prob_maps, axis=0)  # (H, W), tie   
    best_prob = np.max(prob_maps, axis=0)  # (H, W)
    pred_class_map = np.full((H, W), background_id, dtype=np.int64)
    valid = best_prob >= threshold
    pred_class_map[valid] = class_ids[best_idx[valid]]
    return pred_class_map


def to_png_label_dtype(label_map: np.ndarray) -> np.ndarray:
    """
    PNG    dtype  .
    """
    max_label = int(np.max(label_map))
    min_label = int(np.min(label_map))
    if min_label < 0:
        raise ValueError(f"label_map    . min={min_label}")
    if max_label <= np.iinfo(np.uint8).max:
        return label_map.astype(np.uint8)
    if max_label <= np.iinfo(np.uint16).max:
        return label_map.astype(np.uint16)
    raise ValueError(f"PNG  (65535)   . max={max_label}")


def label_map_to_color_overlay(label_map: np.ndarray) -> np.ndarray:
    """
    class map   RGB   (=).
    """
    H, W = label_map.shape
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    labels = np.unique(label_map)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return overlay
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(labels), 1)))[:, :3]
    for i, lb in enumerate(labels):
        overlay[label_map == lb] = colors[i % len(colors)]
    return overlay


def infer_class_map_from_image_and_boxes(
    medsam_model,
    img_np: np.ndarray,
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    mask_threshold: float = 0.5,
) -> dict:
    """
          API.

    Args:
        medsam_model: load_medsam_model(...)  
        img_np: (H,W)  (H,W,3) 
        boxes_xyxy: (N,4)   bbox [x1,y1,x2,y2]
        class_ids: (N,) bbox  ID( )
        mask_threshold: class map  

    Returns:
        dict:
            - pred_class_map (int64)
            - pred_class_map_png (uint8/uint16)
            - prob_maps (N,H,W)
            - masks (N,H,W)
            - boxes_1024 (N,4)
            - image_rgb (H,W,3)
    """
    sample = {
        "image_rgb": img_np,
        "boxes_xyxy": boxes_xyxy,
        "class_ids": class_ids,
        "gt": None,
        "base_name": None,
        "source": "image",
    }
    return run_medsam_inference_on_sample(
        medsam_model=medsam_model,
        sample=sample,
        mask_threshold=mask_threshold,
    )


def run_medsam_inference_on_sample(
    medsam_model,
    sample: dict,
    mask_threshold: float = 0.5,
) -> dict:
    """
    [ ] load_dataset_sample_*   sample dict  class map .
    """
    img_np = sample["image_rgb"]
    boxes_xyxy = np.asarray(sample["boxes_xyxy"], dtype=np.int64)
    class_ids = np.asarray(sample["class_ids"], dtype=np.int64)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
        raise ValueError(f"boxes_xyxy (N,4) .  shape={boxes_xyxy.shape}")
    if class_ids.ndim != 1 or len(class_ids) != len(boxes_xyxy):
        raise ValueError(
            f"class_ids ({len(class_ids)}) bbox ({len(boxes_xyxy)})  ."
        )
    if np.any(class_ids <= 0):
        raise ValueError("class_ids 0   . (0  reserved)")

    device = str(next(medsam_model.parameters()).device)
    img_3c, H, W, img_1024_tensor = _prepare_image_to_1024(img_np, device=device)
    boxes_1024 = (boxes_xyxy / np.array([W, H, W, H], dtype=np.float64) * 1024).astype(np.float32)

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)

    prob_maps, masks = medsam_inference_multi_boxes_with_probs(
        medsam_model=medsam_model,
        img_embed=image_embedding,
        boxes_1024=boxes_1024,
        H=H,
        W=W,
        threshold=mask_threshold,
    )
    pred_class_map = build_prediction_class_map(
        prob_maps=prob_maps,
        class_ids=class_ids,
        threshold=mask_threshold,
        background_id=0,
    )

    return torch.from_numpy(pred_class_map).to(device)




def main():
    parser = argparse.ArgumentParser(
        description="MedSAM inference:    bbox   "
    )
    parser.add_argument("-i", "--data_path", type=str, default="assets/img_demo.png")
    parser.add_argument("-o", "--seg_path", type=str, default="assets/")
    parser.add_argument(
        "--boxes",
        nargs="+",
        type=str,
        default=["[95,255,190,350]"],
        help='bbox . : "[x1,y1,x2,y2]" (--npy_root   )',
    )
    parser.add_argument(
        "--class_ids",
        nargs="+",
        type=int,
        default=None,
        help=" bbox  bbox  ID (: --class_ids 1 3 7)",
    )
    # data/npy  
    parser.add_argument(
        "--npy_root",
        type=str,
        default=None,
        help="data/npy    (: data/npy/CT_Abd).   imgs/gts ",
    )
    parser.add_argument(
        "--npy_name",
        type=str,
        default=None,
        help=" .npy  (--npy_root ).   --npy_index ",
    )
    parser.add_argument(
        "--npy_index",
        type=int,
        default=0,
        help="--npy_root      (0)",
    )
    parser.add_argument(
        "--bbox_shift",
        type=int,
        default=20,
        help="GT bbox     (--npy_root  )",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help=" class map   ",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "-chk", "--checkpoint", type=str, default="work_dir/MedSAM/medsam_vit_b.pth"
    )
    args = parser.parse_args()
    os.makedirs(args.seg_path, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        print(f"[WARN] CUDA    device cpu . (: {device})")
        device = "cpu"
    medsam_model = load_medsam_model(checkpoint=args.checkpoint, device=device)

    use_npy = args.npy_root is not None and os.path.isdir(args.npy_root)

    if use_npy:
        sample = load_dataset_sample_from_npy_root(
            npy_root=args.npy_root,
            npy_index=args.npy_index,
            npy_name=args.npy_name,
            bbox_shift=args.bbox_shift,
        )
    else:
        if args.class_ids is None:
            raise ValueError(
                " bbox  --class_ids   . "
                ': --boxes "[95,255,190,350]" "[200,100,300,280]" --class_ids 1 3'
            )
        sample = load_dataset_sample_from_image(
            data_path=args.data_path,
            boxes_xyxy=_parse_boxes_arg(args.boxes),
            class_ids=np.array(args.class_ids, dtype=np.int64),
        )

    infer_out = run_medsam_inference_on_sample(
        medsam_model=medsam_model,
        sample=sample,
        mask_threshold=args.mask_threshold,
    )
    prob_maps = infer_out["prob_maps"]
    masks = infer_out["masks"]
    pred_class_map = infer_out["pred_class_map"]
    pred_class_map_png = infer_out["pred_class_map_png"]
    img_3c = infer_out["image_rgb"]
    box_np = infer_out["boxes_xyxy"]
    base_name = infer_out["base_name"] or "sample"
    gt = infer_out["gt"]

    N = masks.shape[0]
    unique_classes = np.unique(pred_class_map_png)
    print(
        " bbox : "
        f"{N}, prob_maps shape: {prob_maps.shape}, pred class map shape: {pred_class_map_png.shape}"
    )
    print(
        "pred class map dtype: "
        f"{pred_class_map_png.dtype}, class range: [{int(unique_classes.min())}, {int(unique_classes.max())}], "
        f"unique classes: {unique_classes.tolist()}"
    )

    colors = ["blue", "lime", "red", "orange", "cyan"]
    pred_overlay = label_map_to_color_overlay(pred_class_map)
    if use_npy:
        # 3: +bbox / GT / 
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_3c)
        for i in range(N):
            show_box(box_np[i], axes[0], color=colors[i % len(colors)])
        axes[0].set_title(f"Image + {N} BBox(es)")
        axes[0].axis("off")

        axes[1].imshow(img_3c)
        gt_show = label_map_to_color_overlay(gt)
        axes[1].imshow(gt_show, alpha=0.5)
        axes[1].set_title("GT (from npy)")
        axes[1].axis("off")

        axes[2].imshow(img_3c)
        axes[2].imshow(pred_overlay, alpha=0.5)
        for i in range(N):
            show_box(box_np[i], axes[2], color=colors[i % len(colors)])
        axes[2].set_title("Pred Class Map")
        axes[2].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_3c)
        for i in range(N):
            show_box(box_np[i], axes[0], color=colors[i % len(colors)])
        axes[0].set_title(f"Input Image + {N} Bounding Box(es)")
        axes[0].axis("off")
        axes[1].imshow(img_3c)
        axes[1].imshow(pred_overlay, alpha=0.5)
        for i in range(N):
            show_box(box_np[i], axes[1], color=colors[i % len(colors)])
        axes[1].set_title("Pred Class Map")
        axes[1].axis("off")

    plt.tight_layout()
    vis_path = join(args.seg_path, f"vis_multi_boxes_{base_name}.png")
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    print(f" : {vis_path}")

    pred_map_path = join(args.seg_path, f"pred_class_map_{base_name}.png")
    io.imsave(pred_map_path, pred_class_map_png, check_contrast=False)
    print(f"class map : {pred_map_path}")
    plt.show()


if __name__ == "__main__":
    main()
