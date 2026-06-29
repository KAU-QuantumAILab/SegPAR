import os
import datetime
import argparse
import importlib.util
import numpy as np
import torch
from PIL import Image
from mmseg.apis import init_model, inference_model
from utils import seed_all, save_experiment_results
from attack_module import attack_iterative_save, attack_iterative_save_reg
from dataset import CitySet, ADESet, VOCSet, MedicalNpySet
from evaluation import eval_miou, calculate_l0_norm, calculate_pixel_ratio, calculate_impact
from function import  visualize_segmentation
from tqdm import tqdm
from setproctitle import *
from result_export import export_experiment_results
from MedSAM_Inference_multi_boxes import load_medsam_model, infer_class_map_from_image_and_boxes


def load_config(config_path):
    """
    Load and return config dictionary from a python file at config_path.
    The config file should contain a dictionary named 'config'.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def str_or_none(x):
    return None if x.lower() == "none" else x

def run_experiment(config):
    """
    Main execution function for segmentation model experiments using MMSegmentation
    
    Args:
        config (dict): Configuration dictionary containing model settings and parameters
    """
    # Set initial seed for reproducibility
    seed = 2
    seed_all(seed)

    # Setup device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cuda_device = int(config["device"].split(":")[1])
        torch.cuda.set_device(cuda_device)
    config["device"] = device
    
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
        },
        "CT_Abd":{
            "MedSAM":{
                "checkpoint": 'work_dir/MedSAM/medsam_vit_b.pth'
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

    elif config["model"] == "MedSAM":
        model = load_medsam_model(
        checkpoint=model_cfg["checkpoint"],
        device=config["device"],   
    )

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
        dataset = VOCSet(dataset_dir=config["data_dir"], use_gt=config["use_gt"])
    elif config["dataset"] == "CT_Abd":
        dataset = MedicalNpySet(config["data_dir"], use_gt=config["use_gt"])
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")


    # Record start time
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    print("Start time:", start_timestamp)

    # Setup base directory
    base_dir = os.path.join(config["base_dir"], f"{start_timestamp}_bound_{config['bound']}_use_gt_{config['use_gt']}_reward_type_{config['reward_type']}_model_info_{config['w']}_backbone_{config['backbone']}_lr_{config['RL_learning_rate']}_l0_{config['attack_pixel']*config['bound']*100}_it_max_{config['it_max']}_action_space_{config['action_space']}_attack_type_{config['attack_type']}_update_level_{config['reg_update_level']}")
    print(base_dir)
    # Generate adversarial examples
    if config.get("iterative_save", False):
        #     
        original_images = [img.copy() for img in dataset.images]
        if config["action_space"] == "reg":
            attack_generator = attack_iterative_save_reg(model, dataset, config)
        else:
            attack_generator = attack_iterative_save(model, dataset, config)
        bound_count = 0
        
        try:
            while True:
                bound_count = min(bound_count + config["bound"]/5, config["bound"])
                if config["show_effect"] == True:
                    adv_examples, iteration, dis_positive_list, dis_negative_list = next(attack_generator)
                else:
                    adv_examples, iteration = next(attack_generator)

                
                #      
                save_dir = base_dir

                if bound_count != config["bound"]:  # bound    intermediate  
                    save_dir = os.path.join(base_dir, f"intermediate_bound_{int(bound_count)}")
                    os.makedirs(save_dir, exist_ok=True)

                
                adv_path = os.path.join(save_dir, "adv")
                delta_path = os.path.join(save_dir, "delta")
                adv_seg_path = os.path.join(save_dir, "adv_seg")
                ori_seg_path = os.path.join(save_dir, "ori_seg")

                os.makedirs(adv_path, exist_ok=True)
                os.makedirs(delta_path, exist_ok=True)
                os.makedirs(adv_seg_path, exist_ok=True)
                os.makedirs(ori_seg_path, exist_ok=True)

                #   

                #  
                l0_list = []
                ratio_list = []
                impact_list = []

                for i, name in enumerate(dataset.filenames):
                    name = name.rsplit(".", 1)[0] + ".png"
                    
                    # Create subdirectories if they don't exist
                    adv_subdir = os.path.dirname(os.path.join(adv_path, name))
                    delta_subdir = os.path.dirname(os.path.join(delta_path, name))
                    adv_seg_subdir = os.path.dirname(os.path.join(adv_seg_path, name))
                    ori_seg_subdir = os.path.dirname(os.path.join(ori_seg_path, name))
                    
                    os.makedirs(adv_subdir, exist_ok=True)
                    os.makedirs(delta_subdir, exist_ok=True)
                    os.makedirs(adv_seg_subdir, exist_ok=True)
                    os.makedirs(ori_seg_subdir, exist_ok=True)

                    # Save adversarial example and delta image
                    adv_img = Image.fromarray(adv_examples[i][:, :, ::-1].astype(np.uint8))
                    delta_img = Image.fromarray(np.abs(original_images[i].astype(np.uint8) - adv_examples[i].astype(np.uint8)).astype(np.uint8))
                    adv_img.save(os.path.join(adv_path, name), 'PNG')
                    delta_img.save(os.path.join(delta_path, name), 'PNG')

                    # Save segmentation results
                    if config["dataset"] == "CT_Abd":
                        adv_result = infer_class_map_from_image_and_boxes(model, adv_examples[i], dataset.bboxes[i], dataset.class_ids[i])
                        ori_result = infer_class_map_from_image_and_boxes(model, original_images[i], dataset.bboxes[i], dataset.class_ids[i])

                        visualize_segmentation(original_images[i], ori_result.cpu().numpy(), 
                                            os.path.join(ori_seg_path, name), alpha=0.5, dataset=config["dataset"])
                        visualize_segmentation(adv_examples[i], adv_result.cpu().numpy(), 
                                            os.path.join(adv_seg_path, name), alpha=0.5, dataset=config["dataset"])
                        
                        l0 = calculate_l0_norm(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
                        ratio = calculate_pixel_ratio(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
                        impact = calculate_impact(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8), 
                                            ori_result.cpu().numpy(), 
                                            adv_result.cpu().numpy())

                    else:
                        adv_result = inference_model(model, adv_examples[i])
                        ori_result = inference_model(model, original_images[i])

                        visualize_segmentation(original_images[i], ori_result.pred_sem_seg.data.squeeze().cpu().numpy(), 
                                            os.path.join(ori_seg_path, name), alpha=0.5, dataset=config["dataset"])
                        visualize_segmentation(adv_examples[i], adv_result.pred_sem_seg.data.squeeze().cpu().numpy(), 
                                            os.path.join(adv_seg_path, name), alpha=0.5, dataset=config["dataset"])
                        
                        l0 = calculate_l0_norm(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
                        ratio = calculate_pixel_ratio(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
                        impact = calculate_impact(original_images[i].astype(np.uint8), adv_examples[i].astype(np.uint8), 
                                            ori_result.pred_sem_seg.data.squeeze().cpu().numpy(), 
                                            adv_result.pred_sem_seg.data.squeeze().cpu().numpy())
                    
                    l0_list.append(int(l0))
                    ratio_list.append(float(ratio))
                    impact_list.append(float(impact))

                
                benign_miou_score, adv_miou_score = eval_miou(model, original_images, dataset.gt_images, adv_examples, config)
                
                #  
                

                experimental_results = {
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
                    "adv_miou_score": adv_miou_score
                }
                if config["show_effect"] == True:
                    experimental_results["dis_positive_list"] = dis_positive_list
                    experimental_results["dis_negative_list"] = dis_negative_list
                
                # bound   end_time 
                if bound_count == config["bound"]:
                    experimental_results["end_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                save_experiment_results(experimental_results, config, 
                                     f"{start_timestamp}", 
                                     save_dir=save_dir)
                
                

                if bound_count == config["bound"]:
                    export_experiment_results(base_dir, bound_range=(config["bound"]/5, config["bound"]+1, config["bound"]/5), final_bound=config["bound"], benign_bound=config["bound"]/5)


        except StopIteration:
            print("Attack completed")
    else:
        # Setup output directories for non-iterative case
        adv_path = os.path.join(base_dir, "adv")
        delta_path = os.path.join(base_dir, "delta")
        adv_seg_path = os.path.join(base_dir, "adv_seg")
        ori_seg_path = os.path.join(base_dir, "ori_seg")

        # Create all necessary directories
        os.makedirs(adv_path, exist_ok=True)
        os.makedirs(delta_path, exist_ok=True)
        os.makedirs(adv_seg_path, exist_ok=True)
        os.makedirs(ori_seg_path, exist_ok=True)

        adv_examples, iteration = attack(model, dataset, config)
        
        # Calculate and save final statistics for non-iterative case
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        l0_list = []
        ratio_list = []
        impact_list = []

        for i, name in tqdm(enumerate(dataset.filenames), total=len(dataset.filenames)):
            name = name.rsplit(".", 1)[0] + ".png"
            # Create subdirectories if they don't exist
            adv_subdir = os.path.dirname(os.path.join(adv_path, name))
            delta_subdir = os.path.dirname(os.path.join(delta_path, name))
            adv_seg_subdir = os.path.dirname(os.path.join(adv_seg_path, name))
            ori_seg_subdir = os.path.dirname(os.path.join(ori_seg_path, name))

            os.makedirs(adv_subdir, exist_ok=True)
            os.makedirs(delta_subdir, exist_ok=True)
            os.makedirs(adv_seg_subdir, exist_ok=True)
            os.makedirs(ori_seg_subdir, exist_ok=True)

            # Save adversarial example and delta image
            adv_img = Image.fromarray(adv_examples[i][:, :, ::-1].astype(np.uint8))
            delta_img = Image.fromarray(np.abs(dataset.images[i].astype(np.uint8) - adv_examples[i].astype(np.uint8)).astype(np.uint8))
            adv_img.save(os.path.join(adv_path, name), 'PNG')
            delta_img.save(os.path.join(delta_path, name), 'PNG')

            # Save segmentation results
            adv_result = inference_model(model, adv_examples[i])
            ori_result = inference_model(model, dataset.images[i])

            visualize_segmentation(dataset.images[i], ori_result.pred_sem_seg.data.squeeze().cpu().numpy(), 
                                os.path.join(ori_seg_path, name), alpha=0.5, dataset=config["dataset"])
            visualize_segmentation(adv_examples[i], adv_result.pred_sem_seg.data.squeeze().cpu().numpy(), 
                                os.path.join(adv_seg_path, name), alpha=0.5, dataset=config["dataset"])

            # Calculate metrics
            l0 = calculate_l0_norm(dataset.images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
            ratio = calculate_pixel_ratio(dataset.images[i].astype(np.uint8), adv_examples[i].astype(np.uint8))
            impact = calculate_impact(dataset.images[i].astype(np.uint8), adv_examples[i].astype(np.uint8), 
                                   ori_result.pred_sem_seg.data.squeeze().cpu().numpy(), 
                                   adv_result.pred_sem_seg.data.squeeze().cpu().numpy())
            
            l0_list.append(int(l0))
            ratio_list.append(float(ratio))
            impact_list.append(float(impact))

        benign_miou_score, adv_miou_score = eval_miou(model, original_images, dataset.gt_images, adv_examples, config)

        experimental_results = {
            "start_time": start_timestamp,
            "end_time": end_time.strftime("%Y%m%d_%H%M%S"),
            "elapsed_time": elapsed_time,
            "iteration": iteration.tolist() if torch.is_tensor(iteration) else iteration,
            "iteration_mean": float(iteration.mean()) if torch.is_tensor(iteration) else sum(iteration) / len(iteration),
            "l0": l0_list,
            "l0_mean": sum(l0_list) / len(l0_list),
            "ratio": ratio_list,
            "ratio_mean": sum(ratio_list) / len(ratio_list),
            "impact": impact_list,
            "impact_mean": sum(impact_list) / len(impact_list),
            "benign_miou_score": benign_miou_score,
            "adv_miou_score": adv_miou_score
        }

        save_experiment_results(experimental_results, config, start_timestamp, save_dir=base_dir)

    return experimental_results

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
    parser.add_argument('--iterative_save', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="CUDA device to use (e.g., cuda:0, cuda:1, cuda:2)")
    parser.add_argument("--process_name", type=str, default="seg_attack", help="Process name to display")
    parser.add_argument("--use_gt", type=lambda x: x.lower() == 'true', default=False, help="Whether to use ground truth")
    parser.add_argument("--reward_type", type=str, default="standard", help="Reward type (standard or discrepancy)")
    parser.add_argument("--backbone", type=str, default="conv", help="Backbone type (resnet50 or vit_b16)")
    parser.add_argument("--pretrained", type=lambda x: x.lower() == 'true', default=False, help="Whether to use pretrained model")
    parser.add_argument("--max_inner_it", type=int, default=100, help="Max inner iteration")
    parser.add_argument("--use_lora", type=lambda x: x.lower() == 'true', default=False, help="Whether to use LoRA")
    parser.add_argument("--factor", type=float, default=1, help="Factor value")
    parser.add_argument("--update_valid_mask", type=lambda x: x.lower() == 'true', default=False, help="Whether to update valid mask")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-05, help="RL learning rate value")
    parser.add_argument("--show_effect", type=lambda x: x.lower() == 'true', default=False, help="Whether to show effect")
    parser.add_argument("--update_valid_action", type=lambda x: x.lower() == 'true', default=False, help="Whether to update valid action")
    parser.add_argument("--w", type=float, default=0.5, help="model info value")
    parser.add_argument("--it_max", type=int, default=None, help="Max iteration")
    parser.add_argument("--action_space", type=str, default="standard", help="Action space (standard or reg)")
    parser.add_argument("--attack_type", type=str, default="standard", help="Action type (standard or nearest or grey)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--reg_update_level", default="image", help="reg_update_level")
    parser.add_argument("--delete_attack_pixels", type=lambda x: x.lower() == 'true', default=True, help="Whether to delete attack pixels")
    args = parser.parse_args()
    config = load_config(args.config)

    # Set process title
    setproctitle(args.process_name)

    # Update config with command line arguments
    if args.model is not None:
        config["model"] = args.model
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.num_classes is not None:
        config["num_classes"] = args.num_classes
    if args.attack_pixel is not None:
        config["attack_pixel"] = args.attack_pixel
    if args.bound is not None:
        config["bound"] = int(args.bound)
    if args.patient is not None:
        config["patient"] = int(args.patient)
    if args.iterative_save is not None:
        config["iterative_save"] = args.iterative_save
    if args.cuda_device is not None:
        config["device"] = args.cuda_device
    if args.use_gt is not None:
        config["use_gt"] = args.use_gt
    if args.reward_type is not None:
        config["reward_type"] = args.reward_type
    if args.majority is not None:
        config["majority"] = args.majority
    if args.backbone is not None:
        config["backbone"] = args.backbone
    if args.pretrained is not None:
        config["pretrained"] = args.pretrained
    if args.max_inner_it is not None:
        config["max_inner_it"] = int(args.max_inner_it)
    if args.use_lora is not None:
        config["use_lora"] = args.use_lora
    if args.update_valid_mask is not None:
        config["update_valid_mask"] = args.update_valid_mask
    if args.update_valid_action is not None:
        config["update_valid_action"] = args.update_valid_action
    if args.rl_learning_rate is not None:
        config["RL_learning_rate"] = args.rl_learning_rate
    if args.show_effect is not None:
        config["show_effect"] = args.show_effect
    if args.w is not None:
        config["w"] = args.w
    if args.it_max is not None:
        config["it_max"] = args.it_max
    if args.action_space is not None:
        config["action_space"] = args.action_space
    if args.attack_type is not None:
        config["attack_type"] = args.attack_type
    if args.batch is not None:
        config["batch"] = args.batch
    if args.reg_update_level is not None:
        config["reg_update_level"] = args.reg_update_level
    if args.delete_attack_pixels is not None:
        config["delete_attack_pixels"] = args.delete_attack_pixels
    print(config)
    results = run_experiment(config)
    
    print("Experiment completed successfully")
