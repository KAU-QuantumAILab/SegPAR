import argparse
import datetime
import importlib.util
import os

import numpy as np
import torch
from PIL import Image
from setproctitle import setproctitle
from tqdm import tqdm

from attack_module import adv_setting_attack_iterative_save, adv_setting_attack_iterative_save_reg
from adv_setting import load_model, model_predict
from dataset import ADESet, CitySet, VOCSet
from evaluation import calculate_impact, calculate_l0_norm, calculate_pixel_ratio, eval_miou_adv
from function import visualize_segmentation
from result_export import export_experiment_results
from utils import save_experiment_results, seed_all


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def str_or_none(value):
    return None if value.lower() == "none" else value


def _setup_device(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if ":" in str(device):
            torch.cuda.set_device(int(str(device).split(":")[1]))
        else:
            torch.cuda.set_device(0)
    config["device"] = device


def _load_adv_model(config):
    model = load_model(config)
    checkpoint = torch.load(config["model_path"], map_location=config["device"])

    if config["device"].type == "cuda":
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(config["device"])

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _build_dataset(config):
    if config["dataset"] == "cityscapes":
        return CitySet(dataset_dir=config["data_dir"], use_gt=config["use_gt"])
    if config["dataset"] == "ade20k":
        return ADESet(dataset_dir=config["data_dir"], use_gt=config["use_gt"])
    if config["dataset"] == "VOC2012":
        return VOCSet(dataset_dir=config["data_dir"], use_gt=config["use_gt"])
    raise ValueError(f"Unsupported dataset: {config['dataset']}")


def _save_images_and_metrics(model, original_images, adv_examples, dataset, save_dir, config):
    adv_path = os.path.join(save_dir, "adv")
    delta_path = os.path.join(save_dir, "delta")
    adv_seg_path = os.path.join(save_dir, "adv_seg")
    ori_seg_path = os.path.join(save_dir, "ori_seg")

    os.makedirs(adv_path, exist_ok=True)
    os.makedirs(delta_path, exist_ok=True)
    os.makedirs(adv_seg_path, exist_ok=True)
    os.makedirs(ori_seg_path, exist_ok=True)

    l0_list, ratio_list, impact_list = [], [], []

    for i, name in tqdm(enumerate(dataset.filenames), total=len(dataset.filenames)):
        name = name.rsplit(".", 1)[0] + ".png"

        adv_subdir = os.path.dirname(os.path.join(adv_path, name))
        delta_subdir = os.path.dirname(os.path.join(delta_path, name))
        adv_seg_subdir = os.path.dirname(os.path.join(adv_seg_path, name))
        ori_seg_subdir = os.path.dirname(os.path.join(ori_seg_path, name))

        os.makedirs(adv_subdir, exist_ok=True)
        os.makedirs(delta_subdir, exist_ok=True)
        os.makedirs(adv_seg_subdir, exist_ok=True)
        os.makedirs(ori_seg_subdir, exist_ok=True)

        adv_img = Image.fromarray(adv_examples[i][:, :, ::-1].astype(np.uint8))
        delta_img = Image.fromarray(
            np.abs(original_images[i].astype(np.uint8) - adv_examples[i].astype(np.uint8)).astype(np.uint8)
        )
        adv_img.save(os.path.join(adv_path, name), "PNG")
        delta_img.save(os.path.join(delta_path, name), "PNG")

        _, adv_pred = model_predict(model, adv_examples[i], config)
        _, ori_pred = model_predict(model, original_images[i], config)

        ori_np = ori_pred.cpu().numpy()
        adv_np = adv_pred.cpu().numpy()

        visualize_segmentation(
            original_images[i],
            ori_np,
            os.path.join(ori_seg_path, name),
            alpha=0.5,
            dataset=config["dataset"],
        )
        visualize_segmentation(
            adv_examples[i],
            adv_np,
            os.path.join(adv_seg_path, name),
            alpha=0.5,
            dataset=config["dataset"],
        )

        l0_list.append(int(calculate_l0_norm(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))))
        ratio_list.append(float(calculate_pixel_ratio(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))))
        impact_list.append(float(calculate_impact(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8), ori_np, adv_np)))

    return l0_list, ratio_list, impact_list


def _get_adv_attack_generator(model, dataset, config):
    if config.get("action_space") == "reg":
        return adv_setting_attack_iterative_save_reg(model, dataset, config)
    return adv_setting_attack_iterative_save(model, dataset, config)


def _run_adv_setting_attack(model, dataset, config):
    final_output = None
    for output in _get_adv_attack_generator(model, dataset, config):
        final_output = output
    if final_output is None:
        raise RuntimeError("Attack generator produced no output.")
    return final_output[0], final_output[1]


def run_experiment(config):
    seed_all(2)
    _setup_device(config)

    model = _load_adv_model(config)
    dataset = _build_dataset(config)

    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    base_dir = os.path.join(
        config["base_dir"],
        (
            f"{start_timestamp}_bound_{config['bound']}_use_gt_{config['use_gt']}_"
            f"reward_type_{config['reward_type']}_backbone_{config['backbone']}_"
            f"lr_{config['RL_learning_rate']}_"
            f"l0_{config['attack_pixel'] * config['bound'] * 100}_"
            f"it_max_{config['it_max']}_action_space_{config['action_space']}"
        ),
    )

    original_images = [img.copy() for img in dataset.images]

    if config.get("iterative_save", False):
        attack_generator = _get_adv_attack_generator(model, dataset, config)
        final_results = None
        bound_step = max(1, int(config["bound"] / 5))
        bound_count = 0

        try:
            while True:
                bound_count = min(bound_count + bound_step, config["bound"])
                if config.get("show_effect", False):
                    adv_examples, iteration, dis_positive_list, dis_negative_list = next(attack_generator)
                else:
                    adv_examples, iteration = next(attack_generator)

                save_dir = base_dir
                if bound_count != config["bound"]:
                    save_dir = os.path.join(base_dir, f"intermediate_bound_{bound_count}")
                    os.makedirs(save_dir, exist_ok=True)

                l0_list, ratio_list, impact_list = _save_images_and_metrics(
                    model, original_images, adv_examples, dataset, save_dir, config
                )
                benign_miou_score, adv_miou_score = eval_miou_adv(
                    model, original_images, dataset.gt_images, adv_examples, config
                )

                results = {
                    "start_time": start_timestamp,
                    "current_bound": bound_count,
                    "elapsed_time": (datetime.datetime.now() - start_time).total_seconds(),
                    "iteration": iteration.tolist() if torch.is_tensor(iteration) else iteration,
                    "iteration_mean": float(iteration.mean()) if torch.is_tensor(iteration) else sum(iteration) / len(iteration),
                    "l0": l0_list,
                    "l0_mean": sum(l0_list) / len(l0_list),
                    "ratio": ratio_list,
                    "ratio_mean": sum(ratio_list) / len(ratio_list),
                    "impact": impact_list,
                    "impact_mean": sum(impact_list) / len(impact_list),
                    "benign_miou_score": benign_miou_score,
                    "adv_miou_score": adv_miou_score,
                }
                if config.get("show_effect", False):
                    results["dis_positive_list"] = dis_positive_list
                    results["dis_negative_list"] = dis_negative_list
                if bound_count == config["bound"]:
                    results["end_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                save_experiment_results(results, config, f"{start_timestamp}", save_dir=save_dir)
                final_results = results

                if bound_count == config["bound"]:
                    export_experiment_results(
                        base_dir,
                        bound_range=(bound_step, config["bound"] + 1, bound_step),
                        final_bound=config["bound"],
                        benign_bound=bound_step,
                    )

        except StopIteration:
            print("Attack completed")

        if final_results is None:
            raise RuntimeError("Attack did not produce any result.")

        return final_results

    adv_examples, iteration = _run_adv_setting_attack(model, dataset, config)

    l0_list, ratio_list, impact_list = _save_images_and_metrics(
        model, original_images, adv_examples, dataset, base_dir, config
    )
    benign_miou_score, adv_miou_score = eval_miou_adv(
        model, original_images, dataset.gt_images, adv_examples, config
    )

    results = {
        "start_time": start_timestamp,
        "end_time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "elapsed_time": (datetime.datetime.now() - start_time).total_seconds(),
        "iteration": iteration.tolist() if torch.is_tensor(iteration) else iteration,
        "iteration_mean": float(iteration.mean()) if torch.is_tensor(iteration) else sum(iteration) / len(iteration),
        "l0": l0_list,
        "l0_mean": sum(l0_list) / len(l0_list),
        "ratio": ratio_list,
        "ratio_mean": sum(ratio_list) / len(ratio_list),
        "impact": impact_list,
        "impact_mean": sum(impact_list) / len(impact_list),
        "benign_miou_score": benign_miou_score,
        "adv_miou_score": adv_miou_score,
    }
    save_experiment_results(results, config, start_timestamp, save_dir=base_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, help="Model type (mask2former or segformer)")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--majority", type=str_or_none, help="Majority voting strategy")
    parser.add_argument("--attack_pixel", type=float, help="Attack pixel value")
    parser.add_argument("--bound", type=int, help="Bound value")
    parser.add_argument("--patient", type=int, help="Patient value")
    parser.add_argument('--iterative_save', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="CUDA device to use (e.g., cuda:0, cuda:1, cuda:2)")
    parser.add_argument("--process_name", type=str, default="seg_attack", help="Process name to display")
    parser.add_argument("--use_gt", type=lambda x: x.lower() == 'true', default=False, help="Whether to use ground truth")
    parser.add_argument("--reward_type", type=str, default="standard", help="Reward type (standard or discrepancy)")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone type (resnet50 or vit_b16)")
    parser.add_argument("--pretrained", type=lambda x: x.lower() == 'true', default=False, help="Whether to use pretrained model")
    parser.add_argument("--max_inner_it", type=int, default=100, help="Max inner iteration")
    parser.add_argument("--use_lora", type=lambda x: x.lower() == 'true', default=False, help="Whether to use LoRA")
    parser.add_argument("--factor", type=float, default=1, help="Factor value")
    parser.add_argument("--update_valid_mask", type=lambda x: x.lower() == 'true', default=None, help="Whether to update valid mask")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-05, help="RL learning rate value")
    parser.add_argument("--show_effect", type=lambda x: x.lower() == 'true', default=False, help="Whether to show effect")
    parser.add_argument("--update_valid_action", type=lambda x: x.lower() == 'true', default=False, help="Whether to update valid action")
    parser.add_argument("--w", type=float, default=0.5, help="model info value")
    parser.add_argument("--it_max", type=int, default=None, help="Max iteration")
    parser.add_argument("--action_space", type=str, default="standard", help="Action space (standard or reg)")
    parser.add_argument("--attack_type", default = "standard",  help="Whether to use nearest neighbor")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--delete_attack_pixels", type=lambda x: x.lower() == 'true', default=False, help="Whether to delete attack pixels")
    parser.add_argument("--model_path", type=str)
    

    args = parser.parse_args()
    config = load_config(args.config)

    setproctitle(args.process_name)

    overrides = {
        "model": args.model,
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "attack_pixel": args.attack_pixel,
        "bound": args.bound,
        "patient": args.patient,
        "batch": args.batch,
        "device": args.cuda_device,
        "majority": args.majority,
        "use_gt": args.use_gt,
        "iterative_save": args.iterative_save,
        "reward_type": args.reward_type,
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "max_inner_it": args.max_inner_it,
        "use_lora": args.use_lora,
        "update_valid_mask": args.update_valid_mask,
        "update_valid_action": args.update_valid_action,
        "show_effect": args.show_effect,
        "w": args.w,
        "action_space": args.action_space,
        "attack_type": args.attack_type,
        "delete_attack_pixels": args.delete_attack_pixels,
        "RL_learning_rate": args.rl_learning_rate,
        "it_max": args.it_max,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    print(config)
    run_experiment(config)
    print("Experiment completed successfully")
