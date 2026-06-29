import argparse
import datetime
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dataset import ADESet, VOCSet
from evaluation import (calculate_impact, calculate_l0_norm, calculate_pixel_ratio,
                        eval_miou)
from function import visualize_segmentation
from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from utils import save_experiment_results, seed_all


DATASET_MODEL_CONFIGS = {
    "VOC2012": {
        "deeplabv3": {
            "config": "configs/deeplabv3/deeplabv3_r101-d8_4xb4-20k_voc12aug-512x512.py",
            "checkpoint": "ckpt/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth",
        },
        "pspnet": {
            "config": "configs/pspnet/pspnet_r101-d8_4xb4-20k_voc12aug-512x512.py",
            "checkpoint": "ckpt/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth",
        },
    },
    "ade20k": {
        "deeplabv3": {
            "config": "configs/deeplabv3/deeplabv3_r101-d8_4xb4-160k_ade20k-512x512.py",
            "checkpoint": "ckpt/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth",
        },
        "pspnet": {
            "config": "configs/pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py",
            "checkpoint": "ckpt/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth",
        },
    },
}

DATASET_META = {
    "VOC2012": {
        "num_class": 21,
        "default_data_dir": "datasets/VOC2012",
        "default_base_dir": "results/VOC2012/pgd0",
        "dataset_cls": VOCSet,
    },
    "ade20k": {
        "num_class": 150,
        "default_data_dir": "datasets/ade20k",
        "default_base_dir": "results/ade20k/pgd0",
        "dataset_cls": ADESet,
    },
}


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.lower()
    if key in ("voc", "voc2012"):
        return "VOC2012"
    if key in ("ade", "ade20k"):
        return "ade20k"
    raise ValueError(f"Unsupported dataset: {dataset_name}. Use one of: voc, voc2012, ade, ade20k")


def preprocess_gt_for_loss(y_gt: np.ndarray, dataset_name: str, num_class: int, ignore_index: int = 255):
    """
    Convert dataset GT labels into the index space expected by CE loss.
    Returns (processed_gt, num_clamped_out_of_range_labels).
    """
    y = y_gt.astype(np.int64, copy=True)

    # ADE20K configs in this project use reduce_zero_label=True:
    #   raw 0 -> ignore, raw 1..150 -> 0..149
    if dataset_name == "ade20k":
        y[y == 0] = ignore_index
        valid = y != ignore_index
        y[valid] -= 1

    valid = y != ignore_index
    out_of_range = valid & ((y < 0) | (y >= num_class))
    bad_count = int(out_of_range.sum())
    if bad_count > 0:
        y[out_of_range] = ignore_index

    return y, bad_count


def load_seg_model(dataset_name: str, model_name: str, device: torch.device):
    if dataset_name not in DATASET_MODEL_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if model_name not in DATASET_MODEL_CONFIGS[dataset_name]:
        raise ValueError(f"Unsupported model '{model_name}' for dataset '{dataset_name}'")
    model_cfg = DATASET_MODEL_CONFIGS[dataset_name][model_name]
    model = init_model(model_cfg["config"], None, device=str(device))
    checkpoint = torch.load(model_cfg["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_mmseg_logits(model, x_nhwc: torch.Tensor) -> torch.Tensor:
    n, h, w, _ = x_nhwc.shape
    data_samples = []
    chw_inputs = []
    for i in range(n):
        x_chw = x_nhwc[i].permute(2, 0, 1).contiguous()
        sample = SegDataSample()
        sample.set_metainfo(
            dict(
                ori_shape=(h, w),
                img_shape=(h, w),
                pad_shape=(h, w),
                padding_size=[0, 0, 0, 0],
            )
        )
        data_samples.append(sample)
        chw_inputs.append(x_chw)

    processed = model.data_preprocessor(
        {"inputs": chw_inputs, "data_samples": data_samples},
        training=False,
    )
    batch_inputs = processed["inputs"]
    batch_metas = [sample.metainfo for sample in processed["data_samples"]]
    return model.inference(batch_inputs, batch_metas)


def model_correct_prediction_and_grad(
    model,
    x_nhwc: np.ndarray,
    y_nhw: np.ndarray,
    device: torch.device,
    correct_threshold: float,
):
    x = torch.from_numpy(np.ascontiguousarray(x_nhwc.astype(np.float32, copy=False))).to(device)
    x.requires_grad_(True)
    y = torch.from_numpy(np.ascontiguousarray(y_nhw.astype(np.int64, copy=False))).to(device)

    logits = build_mmseg_logits(model, x)
    ce_map = F.cross_entropy(logits, y, ignore_index=255, reduction="none")
    valid = y != 255
    valid_count = torch.clamp(valid.sum(dim=(1, 2)), min=1).float()
    loss_per_sample = (ce_map * valid.float()).sum(dim=(1, 2)) / valid_count
    loss = loss_per_sample.mean()
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        acc = (((pred == y) & valid).sum(dim=(1, 2)).float()) / valid_count
        corr_pred = (acc >= correct_threshold).detach().cpu().numpy().astype(bool)

    return (
        corr_pred,
        grad.detach().cpu().numpy(),
        acc.detach().cpu().numpy(),
        loss_per_sample.detach().cpu().numpy(),
    )


def model_correct_prediction_and_pixel_acc(
    model,
    x_nhwc: np.ndarray,
    y_nhw: np.ndarray,
    device: torch.device,
    correct_threshold: float,
):
    x = torch.from_numpy(np.ascontiguousarray(x_nhwc.astype(np.float32, copy=False))).to(device)
    y = torch.from_numpy(np.ascontiguousarray(y_nhw.astype(np.int64, copy=False))).to(device)

    with torch.no_grad():
        logits = build_mmseg_logits(model, x)
        ce_map = F.cross_entropy(logits, y, ignore_index=255, reduction="none")
        valid = y != 255
        valid_count = torch.clamp(valid.sum(dim=(1, 2)), min=1).float()
        loss_per_sample = (ce_map * valid.float()).sum(dim=(1, 2)) / valid_count
        pred = torch.argmax(logits, dim=1)
        acc = (((pred == y) & valid).sum(dim=(1, 2)).float()) / valid_count
        corr_pred = (acc >= correct_threshold).detach().cpu().numpy().astype(bool)
        pixel_acc = acc.detach().cpu().numpy()

    return corr_pred, pixel_acc, loss_per_sample.detach().cpu().numpy()


def project_L0_box(y, k, lb, ub):
    """ projection of the batch y to a batch x such that:
          - each image of the batch x has at most k pixels with non-zero channels
          - lb <= x <= ub """

    x = np.copy(y)
    p1 = np.sum(x**2, axis=-1)
    p2 = np.minimum(np.minimum(ub - x, x - lb), 0)
    p2 = np.sum(p2**2, axis=-1)
    p3 = np.sort(np.reshape(p1 - p2, [p2.shape[0], -1]))[:, -k]
    x = x * (np.logical_and(lb <= x, x <= ub)) + lb * (lb > x) + ub * (x > ub)
    x *= np.expand_dims((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)

    return x


def project_L0_sigma(y, k, sigma, kappa, x_nat):
    """ projection of the batch y to a batch x such that:
          - 0 <= x <= 1
          - each image of the batch x differs from the corresponding one of
            x_nat in at most k pixels
          - (1 - kappa*sigma)*x_nat <= x <= (1 + kappa*sigma)*x_nat """

    x = np.copy(y)
    p1 = 1.0 / np.maximum(1e-12, sigma) * (x_nat > 0).astype(float) + 1e12 * (x_nat == 0).astype(float)
    p2 = (
        1.0
        / np.maximum(1e-12, sigma)
        * (1.0 / np.maximum(1e-12, x_nat) - 1)
        * (x_nat > 0).astype(float)
        + 1e12 * (x_nat == 0).astype(float)
        + 1e12 * (sigma == 0).astype(float)
    )
    lmbd_l = np.maximum(-kappa, np.amax(-p1, axis=-1, keepdims=True))
    lmbd_u = np.minimum(kappa, np.amin(p2, axis=-1, keepdims=True))

    lmbd_unconstr = np.sum((y - x_nat) * sigma * x_nat, axis=-1, keepdims=True) / np.maximum(
        1e-12, np.sum((sigma * x_nat) ** 2, axis=-1, keepdims=True)
    )
    lmbd = np.maximum(lmbd_l, np.minimum(lmbd_unconstr, lmbd_u))

    p12 = np.sum((y - x_nat) ** 2, axis=-1, keepdims=True)
    p22 = np.sum((y - (1 + lmbd * sigma) * x_nat) ** 2, axis=-1, keepdims=True)
    p3 = np.sort(np.reshape(p12 - p22, [x.shape[0], -1]))[:, -k]

    x = x_nat + lmbd * sigma * x_nat * ((p12 - p22) >= p3.reshape([-1, 1, 1, 1]))

    return x


def perturb_L0_box(attack, x_nat, y_nat, lb, ub, model, device):
    """ PGD attack wrt L0-norm + box constraints """

    if attack.rs:
        x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
        x2 = np.clip(x2, 0, 255)
    else:
        x2 = np.copy(x_nat)

    adv_not_found = np.ones((x_nat.shape[0],), dtype=np.int64)
    adv = np.copy(x_nat)

    for i in range(attack.num_steps):
        if i > 0:
            pred, grad, _, _ = model_correct_prediction_and_grad(
                model=model,
                x_nhwc=x2,
                y_nhw=y_nat,
                device=device,
                correct_threshold=attack.correct_threshold,
            )
            pred_int = pred.astype(np.int64)
            adv_not_found = np.minimum(adv_not_found, pred_int)
            adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

            grad /= 1e-10 + np.sum(np.abs(grad), axis=(1, 2, 3), keepdims=True)
            x2 = np.add(x2, (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + attack.step_size * grad, casting="unsafe")

        x2 = x_nat + project_L0_box(x2 - x_nat, attack.k, lb, ub)

    return adv, adv_not_found


def perturb_L0_sigma(attack, x_nat, y_nat, model, device):
    """ PGD attack wrt L0-norm + sigma-map constraints """

    if attack.rs:
        x2 = x_nat + np.random.uniform(-attack.kappa, attack.kappa, x_nat.shape)
        x2 = np.clip(x2, 0, 255)
    else:
        x2 = np.copy(x_nat)

    adv_not_found = np.ones((x_nat.shape[0],), dtype=np.int64)
    adv = np.copy(x_nat)

    for i in range(attack.num_steps):
        if i > 0:
            pred, grad, _, _ = model_correct_prediction_and_grad(
                model=model,
                x_nhwc=x2,
                y_nhw=y_nat,
                device=device,
                correct_threshold=attack.correct_threshold,
            )
            pred_int = pred.astype(np.int64)
            adv_not_found = np.minimum(adv_not_found, pred_int)
            adv[np.logical_not(pred)] = np.copy(x2[np.logical_not(pred)])

            grad /= 1e-10 + np.sum(np.abs(grad), axis=(1, 2, 3), keepdims=True)
            x2 = np.add(x2, (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + attack.step_size * grad, casting="unsafe")

        x2 = project_L0_sigma(x2, attack.k, attack.sigma, attack.kappa, x_nat)

    return adv, adv_not_found


def sigma_map(x):
    """ creates the sigma-map for the batch x """

    sh = [4]
    sh.extend(x.shape)
    t = np.zeros(sh)
    t[0, :, :-1] = x[:, 1:]
    t[0, :, -1] = x[:, -1]
    t[1, :, 1:] = x[:, :-1]
    t[1, :, 0] = x[:, 0]
    t[2, :, :, :-1] = x[:, :, 1:]
    t[2, :, :, -1] = x[:, :, -1]
    t[3, :, :, 1:] = x[:, :, :-1]
    t[3, :, :, 0] = x[:, :, 0]

    mean1 = (t[0] + x + t[1]) / 3
    sd1 = np.sqrt(((t[0] - mean1) ** 2 + (x - mean1) ** 2 + (t[1] - mean1) ** 2) / 3)

    mean2 = (t[2] + x + t[3]) / 3
    sd2 = np.sqrt(((t[2] - mean2) ** 2 + (x - mean2) ** 2 + (t[3] - mean2) ** 2) / 3)

    sd = np.minimum(sd1, sd2)
    sd = np.sqrt(sd)

    return sd


class PGDattack:
    def __init__(self, model, args, device):
        self.model = model
        self.device = device
        self.type_attack = args["type_attack"]  # 'L0', 'L0+Linf', 'L0+sigma'
        self.num_steps = args["num_steps"]
        self.step_size = args["step_size"]
        self.n_restarts = args["n_restarts"]
        self.rs = args.get("rs", True)
        self.epsilon = args["epsilon"]
        self.kappa = args["kappa"]
        self.k = args["sparsity"]
        self.correct_threshold = args.get("correct_threshold", 0.99)

    def perturb(self, x_nat, y_nat):
        adv = np.copy(x_nat)

        if self.type_attack == "L0+sigma":
            self.sigma = sigma_map(x_nat)

        for counter in range(self.n_restarts):
            if counter == 0:
                corr_pred, _, _, _ = model_correct_prediction_and_grad(
                    model=self.model,
                    x_nhwc=x_nat,
                    y_nhw=y_nat,
                    device=self.device,
                    correct_threshold=self.correct_threshold,
                )
                pgd_adv_acc = np.copy(corr_pred.astype(np.int64))

            if self.type_attack == "L0":
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(self, x_nat, y_nat, -x_nat, 255.0 - x_nat, self.model, self.device)
            elif self.type_attack == "L0+Linf":
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(
                    self,
                    x_nat,
                    y_nat,
                    np.maximum(-self.epsilon, -x_nat),
                    np.minimum(self.epsilon, 255.0 - x_nat),
                    self.model,
                    self.device,
                )
            elif self.type_attack == "L0+sigma" and x_nat.shape[3] == 3:
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_sigma(self, x_nat, y_nat, self.model, self.device)
            elif self.type_attack == "L0+sigma" and x_nat.shape[3] == 1:
                x_batch_adv, curr_pgd_adv_acc = perturb_L0_box(
                    self,
                    x_nat,
                    y_nat,
                    np.maximum(-self.kappa * self.sigma, -x_nat),
                    np.minimum(self.kappa * self.sigma, 255.0 - x_nat),
                    self.model,
                    self.device,
                )
            else:
                raise ValueError(f"Unsupported type_attack: {self.type_attack}")

            pgd_adv_acc = np.minimum(pgd_adv_acc, curr_pgd_adv_acc)

            print(f"Restart {counter + 1} - Robust accuracy: {np.sum(pgd_adv_acc) / x_nat.shape[0]:.6f}")
            adv[np.logical_not(curr_pgd_adv_acc.astype(bool))] = x_batch_adv[np.logical_not(curr_pgd_adv_acc.astype(bool))]

        pixels_changed = np.sum(np.amax(np.abs(adv - x_nat) > 1e-10, axis=-1), axis=(1, 2))
        print("Pixels changed:", pixels_changed)
        corr_pred, _, _, _ = model_correct_prediction_and_grad(
            model=self.model,
            x_nhwc=adv,
            y_nhw=y_nat,
            device=self.device,
            correct_threshold=self.correct_threshold,
        )
        print(f"Robust accuracy at {self.k} pixels: {np.sum(corr_pred) / x_nat.shape[0] * 100.0:.2f}%")
        print(f"Maximum perturbation size: {np.amax(np.abs(adv - x_nat)):.5f}")

        return adv, pgd_adv_acc


def save_checkpoint(
    model,
    config: Dict,
    dataset,
    original_images: List[np.ndarray],
    adv_images_float: List[np.ndarray],
    iteration: int,
    start_time: datetime.datetime,
    start_timestamp: str,
    base_dir: str,
    max_iter: int,
) -> None:
    save_dir = base_dir if iteration == max_iter else os.path.join(base_dir, f"intermediate_bound_{iteration}")
    os.makedirs(save_dir, exist_ok=True)

    adv_path = os.path.join(save_dir, "adv")
    delta_path = os.path.join(save_dir, "delta")
    adv_seg_path = os.path.join(save_dir, "adv_seg")
    ori_seg_path = os.path.join(save_dir, "ori_seg")
    os.makedirs(adv_path, exist_ok=True)
    os.makedirs(delta_path, exist_ok=True)
    os.makedirs(adv_seg_path, exist_ok=True)
    os.makedirs(ori_seg_path, exist_ok=True)

    adv_examples = [np.clip(np.round(x), 0, 255).astype(np.uint8) for x in adv_images_float]
    l0_list: List[int] = []
    ratio_list: List[float] = []
    impact_list: List[float] = []

    for i, name in tqdm(enumerate(dataset.filenames), total=len(dataset.filenames), desc=f"Saving @ {iteration}"):
        name = name.rsplit(".", 1)[0] + ".png"
        adv_file = os.path.join(adv_path, name)
        delta_file = os.path.join(delta_path, name)
        adv_seg_file = os.path.join(adv_seg_path, name)
        ori_seg_file = os.path.join(ori_seg_path, name)

        os.makedirs(os.path.dirname(adv_file), exist_ok=True)
        os.makedirs(os.path.dirname(delta_file), exist_ok=True)
        os.makedirs(os.path.dirname(adv_seg_file), exist_ok=True)
        os.makedirs(os.path.dirname(ori_seg_file), exist_ok=True)

        adv_img = Image.fromarray(adv_examples[i][:, :, ::-1].astype(np.uint8))
        delta_img = Image.fromarray(np.abs(original_images[i].astype(np.uint8) - adv_examples[i].astype(np.uint8)).astype(np.uint8))
        adv_img.save(adv_file, "PNG")
        delta_img.save(delta_file, "PNG")

        adv_result = inference_model(model, adv_examples[i])
        ori_result = inference_model(model, original_images[i])
        ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy()
        adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy()

        visualize_segmentation(original_images[i], ori_pred, ori_seg_file, alpha=0.5, dataset=config["dataset"])
        visualize_segmentation(adv_examples[i], adv_pred, adv_seg_file, alpha=0.5, dataset=config["dataset"])

        l0 = calculate_l0_norm(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
        ratio = calculate_pixel_ratio(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
        impact = calculate_impact(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8), ori_pred, adv_pred)
        l0_list.append(int(l0))
        ratio_list.append(float(ratio))
        impact_list.append(float(impact))

    benign_miou_score, adv_miou_score = eval_miou(model, original_images, dataset.gt_images, adv_examples, config)

    experimental_results = {
        "start_time": start_timestamp,
        "current_bound": iteration,
        "elapsed_time": (datetime.datetime.now() - start_time).total_seconds(),
        "iteration": [iteration] * len(dataset.filenames),
        "iteration_mean": float(iteration),
        "l0": l0_list,
        "l0_mean": float(sum(l0_list) / len(l0_list)),
        "ratio": ratio_list,
        "ratio_mean": float(sum(ratio_list) / len(ratio_list)),
        "impact": impact_list,
        "impact_mean": float(sum(impact_list) / len(impact_list)),
        "benign_miou_score": benign_miou_score,
        "adv_miou_score": adv_miou_score,
    }

    if iteration == max_iter:
        experimental_results["end_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_experiment_results(experimental_results, config, f"{start_timestamp}", save_dir=save_dir)


def print_run_config(config: Dict[str, object]) -> None:
    print("Run Config")
    print("========================================")
    for key in sorted(config.keys()):
        print(f"{key}: {config[key]}")
    print("========================================")


def run_pgd0_voc(args):
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.n_restarts < 1:
        raise ValueError(f"n_restarts must be >= 1, got {args.n_restarts}")
    if args.restart_batch_size < 1:
        raise ValueError(f"restart_batch_size must be >= 1, got {args.restart_batch_size}")
    restart_batch_size = min(args.restart_batch_size, args.n_restarts)

    dataset_name = normalize_dataset_name(args.dataset)
    dataset_meta = DATASET_META[dataset_name]
    data_dir = args.data_dir if args.data_dir is not None else dataset_meta["default_data_dir"]
    base_dir_root = args.base_dir if args.base_dir is not None else dataset_meta["default_base_dir"]

    config = {
        "task": "segmentation",
        "dataset": dataset_name,
        "model": args.model,
        "data_dir": data_dir,
        "base_dir": base_dir_root,
        "num_class": dataset_meta["num_class"],
        "attack_type": args.type_attack,
        "attack_pixel": args.attack_pixel,
        "bound": args.max_iter,
        "save_interval": args.save_interval,
        "n_restarts": args.n_restarts,
        "step_size": args.step_size,
        "device": str(device),
    }
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(
        base_dir_root,
        f"{start_timestamp}_bound_{args.max_iter}_use_gt_True_reward_type_{args.type_attack}_model_{args.model}_l0_{args.attack_pixel*args.max_iter*100}",
    )
    config["output_dir"] = base_dir
    config["input_value_range"] = "0..255"
    print_run_config(config)

    model = load_seg_model(dataset_name, args.model, device)
    dataset = dataset_meta["dataset_cls"](dataset_dir=data_dir, use_gt=True)

    original_images = [img.copy() for img in dataset.images]
    x_nat_list = [img.astype(np.float32) for img in original_images]
    y_nat_list = []
    total_bad_labels = 0
    for gt in dataset.gt_images:
        y_proc, bad_count = preprocess_gt_for_loss(
            y_gt=gt,
            dataset_name=dataset_name,
            num_class=dataset_meta["num_class"],
            ignore_index=255,
        )
        y_nat_list.append(y_proc)
        total_bad_labels += bad_count
    if total_bad_labels > 0:
        print(
            f"[warning] Clamped {total_bad_labels} out-of-range GT labels to ignore_index=255 "
            f"for dataset={dataset_name}"
        )

    def maybe_update_best(state, candidate_x, loss_value: float, pixel_acc: float, is_correct: bool):
        best_loss = state["best_loss"]
        better = loss_value > best_loss
        tie_break = np.isclose(loss_value, best_loss) and (
            ((not is_correct) and state["best_is_correct"])
            or ((bool(is_correct) == bool(state["best_is_correct"])) and (pixel_acc < state["best_pixel_acc"]))
        )
        if better or tie_break:
            state["best_loss"] = float(loss_value)
            state["best_pixel_acc"] = float(pixel_acc)
            state["best_x2"] = np.copy(candidate_x)
            state["best_is_correct"] = bool(is_correct)

    attack_states = []
    for i, x_nat in enumerate(x_nat_list):
        h, w = x_nat.shape[:2]
        k = max(1, int(h * w * args.attack_pixel))
        lb = -x_nat
        ub = 255.0 - x_nat
        sigma = sigma_map(x_nat[None]) if args.type_attack == "L0+sigma" else None

        if args.type_attack == "L0":
            proj_lb = lb
            proj_ub = ub
        elif args.type_attack == "L0+Linf":
            proj_lb = np.maximum(-args.epsilon, lb)
            proj_ub = np.minimum(args.epsilon, ub)
        elif args.type_attack == "L0+sigma":
            if x_nat.shape[2] == 3:
                proj_lb = None
                proj_ub = None
            elif x_nat.shape[2] == 1:
                proj_lb = np.maximum(-args.kappa * sigma[0], lb)
                proj_ub = np.minimum(args.kappa * sigma[0], ub)
            else:
                raise ValueError(f"Unsupported channel count for L0+sigma: {x_nat.shape[2]}")
        else:
            raise ValueError(f"Unsupported type_attack: {args.type_attack}")

        restart_x2 = np.zeros((args.n_restarts, *x_nat.shape), dtype=np.float32)
        for restart_idx in range(args.n_restarts):
            if args.random_start:
                x2 = x_nat + np.random.uniform(lb, ub, x_nat.shape)
                x2 = np.clip(x2, 0, 255)
            else:
                x2 = np.copy(x_nat)

            x2_batch = x2[None]
            if args.type_attack in ("L0", "L0+Linf"):
                x2_batch = x_nat[None] + project_L0_box(x2_batch - x_nat[None], k, proj_lb[None], proj_ub[None])
            elif args.type_attack == "L0+sigma" and x_nat.shape[2] == 3:
                x2_batch = project_L0_sigma(x2_batch, k, sigma, args.kappa, x_nat[None])
            elif args.type_attack == "L0+sigma" and x_nat.shape[2] == 1:
                x2_batch = x_nat[None] + project_L0_box(x2_batch - x_nat[None], k, proj_lb[None], proj_ub[None])
            else:
                raise ValueError(f"Unsupported type_attack: {args.type_attack}")

            restart_x2[restart_idx] = np.clip(x2_batch[0], 0, 255).astype(np.float32, copy=False)

        attack_states.append(
            {
                "x_nat": x_nat,
                "y_nat": y_nat_list[i],
                "k": k,
                "lb": lb,
                "ub": ub,
                "sigma": sigma,
                "proj_lb": proj_lb,
                "proj_ub": proj_ub,
                "restart_x2": restart_x2,
                "best_x2": np.copy(restart_x2[0]),
                "best_loss": float("-inf"),
                "best_pixel_acc": float("inf"),
                "best_is_correct": True,
            }
        )

    initial_loss_values = []
    for state in attack_states:
        for chunk_start in range(0, args.n_restarts, restart_batch_size):
            chunk_end = min(args.n_restarts, chunk_start + restart_batch_size)
            x_batch = state["restart_x2"][chunk_start:chunk_end]
            y_batch = np.repeat(state["y_nat"][None], chunk_end - chunk_start, axis=0)
            corr_pred, pixel_acc, loss_values = model_correct_prediction_and_pixel_acc(
                model=model,
                x_nhwc=x_batch,
                y_nhw=y_batch,
                device=device,
                correct_threshold=args.correct_threshold,
            )

            initial_loss_values.extend(loss_values.tolist())
            for local_idx, restart_idx in enumerate(range(chunk_start, chunk_end)):
                maybe_update_best(
                    state=state,
                    candidate_x=state["restart_x2"][restart_idx],
                    loss_value=float(loss_values[local_idx]),
                    pixel_acc=float(pixel_acc[local_idx]),
                    is_correct=bool(corr_pred[local_idx]),
                )

    for step_idx in range(args.max_iter):
        it = step_idx + 1
        iter_loss_values = []
        if step_idx > 0:
            for state in attack_states:
                for chunk_start in range(0, args.n_restarts, restart_batch_size):
                    chunk_end = min(args.n_restarts, chunk_start + restart_batch_size)
                    x_batch = state["restart_x2"][chunk_start:chunk_end]
                    y_batch = np.repeat(state["y_nat"][None], chunk_end - chunk_start, axis=0)

                    pred, grad, pixel_acc, loss_values = model_correct_prediction_and_grad(
                        model=model,
                        x_nhwc=x_batch,
                        y_nhw=y_batch,
                        device=device,
                        correct_threshold=args.correct_threshold,
                    )
                    iter_loss_values.extend(loss_values.tolist())

                    for local_idx, restart_idx in enumerate(range(chunk_start, chunk_end)):
                        maybe_update_best(
                            state=state,
                            candidate_x=state["restart_x2"][restart_idx],
                            loss_value=float(loss_values[local_idx]),
                            pixel_acc=float(pixel_acc[local_idx]),
                            is_correct=bool(pred[local_idx]),
                        )

                    grad /= 1e-10 + np.sum(np.abs(grad), axis=(1, 2, 3), keepdims=True)
                    x2 = np.add(
                        x_batch,
                        (np.random.random_sample(grad.shape) - 0.5) * 1e-12 + args.step_size * grad,
                        casting="unsafe",
                    )

                    if args.type_attack in ("L0", "L0+Linf"):
                        x2 = state["x_nat"][None] + project_L0_box(
                            x2 - state["x_nat"][None],
                            state["k"],
                            state["proj_lb"][None],
                            state["proj_ub"][None],
                        )
                    elif args.type_attack == "L0+sigma" and state["x_nat"].shape[2] == 3:
                        x2 = project_L0_sigma(x2, state["k"], state["sigma"], args.kappa, state["x_nat"][None])
                    elif args.type_attack == "L0+sigma" and state["x_nat"].shape[2] == 1:
                        x2 = state["x_nat"][None] + project_L0_box(
                            x2 - state["x_nat"][None],
                            state["k"],
                            state["proj_lb"][None],
                            state["proj_ub"][None],
                        )
                    else:
                        raise ValueError(f"Unsupported type_attack: {args.type_attack}")

                    state["restart_x2"][chunk_start:chunk_end] = np.clip(x2, 0, 255).astype(np.float32, copy=False)
        else:
            iter_loss_values = list(initial_loss_values)

        mean_iter_loss = float(np.mean(iter_loss_values))
        mean_best_loss = float(np.mean([state["best_loss"] for state in attack_states]))
        mean_best_pixel_acc = float(np.mean([state["best_pixel_acc"] for state in attack_states]))
        best_robust_acc = float(np.mean([1.0 if state["best_is_correct"] else 0.0 for state in attack_states]))
        print(
            f"[iter {it}/{args.max_iter}] mean_loss={mean_iter_loss:.6f}, "
            f"best_loss={mean_best_loss:.6f}, best_pixel_acc={mean_best_pixel_acc:.6f}, "
            f"best_robust_acc={best_robust_acc:.6f}"
        )

        if it % args.save_interval == 0 or it == args.max_iter:
            for state in attack_states:
                for chunk_start in range(0, args.n_restarts, restart_batch_size):
                    chunk_end = min(args.n_restarts, chunk_start + restart_batch_size)
                    x_batch = state["restart_x2"][chunk_start:chunk_end]
                    y_batch = np.repeat(state["y_nat"][None], chunk_end - chunk_start, axis=0)
                    corr_pred, pixel_acc, loss_values = model_correct_prediction_and_pixel_acc(
                        model=model,
                        x_nhwc=x_batch,
                        y_nhw=y_batch,
                        device=device,
                        correct_threshold=args.correct_threshold,
                    )
                    for local_idx, restart_idx in enumerate(range(chunk_start, chunk_end)):
                        maybe_update_best(
                            state=state,
                            candidate_x=state["restart_x2"][restart_idx],
                            loss_value=float(loss_values[local_idx]),
                            pixel_acc=float(pixel_acc[local_idx]),
                            is_correct=bool(corr_pred[local_idx]),
                        )

            adv_images_float = [state["best_x2"] for state in attack_states]
            save_checkpoint(
                model=model,
                config=config,
                dataset=dataset,
                original_images=original_images,
                adv_images_float=adv_images_float,
                iteration=it,
                start_time=start_time,
                start_timestamp=start_timestamp,
                base_dir=base_dir,
                max_iter=args.max_iter,
            )

    print(f"PGD attack completed. Results saved under: {base_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch PGD attack for VOC2012/ADE20K segmentation")
    parser.add_argument("--dataset", type=str, default="VOC2012", help="voc|voc2012|ade|ade20k")
    parser.add_argument("--model", type=str, default="deeplabv3", choices=["deeplabv3", "pspnet"])
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root directory (auto by --dataset if omitted)")
    parser.add_argument("--base_dir", type=str, default=None, help="Result base directory (auto by --dataset if omitted)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--attack_pixel", type=float, default=4e-4)
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--n_restarts", type=int, default=1)
    parser.add_argument("--restart_batch_size", type=int, default=4, help="Number of restarts to process per forward/backward")
    parser.add_argument("--type_attack", type=str, default="L0", choices=["L0", "L0+Linf", "L0+sigma"])
    parser.add_argument("--epsilon", type=float, default=-1)
    parser.add_argument("--kappa", type=float, default=0.8)
    parser.add_argument("--correct_threshold", type=float, default=0.99)
    parser.add_argument("--random_start", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run_pgd0_voc(parse_args())
