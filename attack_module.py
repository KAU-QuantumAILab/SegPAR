# attack_module.py

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import Environment
import Adversarial_RL_simple
from Adversarial_RL_simple import get_resnet_backbone, loraize_cnn, load_agent
from utils import seed_all, sample_action, early_stopping, update
from function import data_preprocessing, StateProcessor, prepare_agent_input, apply_batch_binary_attacks, AttackDataset
from dataset import CitySet
from tqdm import tqdm
from mmseg.apis import inference_model
import copy
from adv_setting import model_predict as adv_model_predict
from MedSAM_Inference_multi_boxes import infer_class_map_from_image_and_boxes

def _init_class_reward_memory(train_data_attack_set):
    """
         reward memory .
    """
    return [
        torch.zeros(len(sample["meta"]), dtype=torch.float32)
        for sample in train_data_attack_set
    ]


def _init_class_perturb_memory(train_data_attack_set):
    """
      (perturbation)  .
    """
    return [
        [None for _ in sample["meta"]]
        for sample in train_data_attack_set
    ]


def _clone_action_data(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return np.array(value, copy=True)


def _align_actions_with_meta(xy_coords, rgb_actions, meta_list):
    """
    RL_reg     meta_list (/ ) .
    """
    aligned = []
    ptr = 0
    for image_meta in meta_list:
        per_image_actions = []
        for obj_meta in image_meta:
            if int(obj_meta.get("attack_pixels", 0)) <= 0:
                per_image_actions.append(None)
                continue
            if ptr >= len(xy_coords) or ptr >= len(rgb_actions):
                per_image_actions.append(None)
                continue
            per_image_actions.append(
                {
                    "coords": _clone_action_data(xy_coords[ptr]),
                    "rgb_actions": _clone_action_data(rgb_actions[ptr]),
                }
            )
            ptr += 1
        aligned.append(per_image_actions)
    return aligned


def _update_class_memories_from_batch(
    rewards,
    start_idx,
    meta_list,
    aligned_actions,
    class_reward_memory,
    class_perturb_memory,
    update_rewards_indices,
    update_rewards_memory,
):
    """
     reward    (/perturbation) .
    """
    reward_ptr = 0
    for local_img_idx, image_meta in enumerate(meta_list):
        global_img_idx = start_idx + local_img_idx
        class_count = len(image_meta)
        if class_count <= 0:
            continue

        end_ptr = min(reward_ptr + class_count, rewards.numel())
        current = rewards[reward_ptr:end_ptr]
        reward_ptr = end_ptr

        if current.numel() < class_count:
            pad = torch.zeros(class_count - current.numel(), dtype=rewards.dtype)
            current = torch.cat((current, pad), dim=0)

        prev = class_reward_memory[global_img_idx]
        if prev.numel() != class_count:
            resized_prev = torch.zeros(class_count, dtype=current.dtype)
            copy_count = min(prev.numel(), class_count)
            if copy_count > 0:
                resized_prev[:copy_count] = prev[:copy_count]
            prev = resized_prev

        improved = current > prev
        if improved.any():
            update_rewards_indices[global_img_idx] = True
            prev = torch.where(improved, current, prev)
            for cls_idx, is_improved in enumerate(improved.tolist()):
                if not is_improved:
                    continue
                action_data = aligned_actions[local_img_idx][cls_idx]
                if action_data is not None:
                    class_perturb_memory[global_img_idx][cls_idx] = action_data

        class_reward_memory[global_img_idx] = prev
        update_rewards_memory[global_img_idx] = prev.mean() if prev.numel() > 0 else update_rewards_memory[global_img_idx]


def _compose_images_from_class_perturb_memory(base_images, train_data_attack_set, class_perturb_memory, config):
    """
     perturbation    .
    """
    modified_images = []
    for img_idx, base_image in enumerate(base_images):
        sample = train_data_attack_set[img_idx]
        meta = sample["meta"]
        masks = sample["mask"]

        action_coords = []
        action_values = []
        selected_meta = []
        selected_masks = []

        for cls_idx, perturb in enumerate(class_perturb_memory[img_idx]):
            if perturb is None:
                continue
            if cls_idx >= len(meta):
                continue
            if int(meta[cls_idx].get("attack_pixels", 0)) <= 0:
                continue

            action_coords.append(perturb["coords"])
            action_values.append(perturb["rgb_actions"])
            selected_meta.append(meta[cls_idx])
            selected_masks.append(masks[cls_idx])

        if len(action_coords) == 0:
            modified_images.append(base_image.copy())
            continue

        selected_masks = torch.stack(selected_masks, dim=0)
        rebuilt = apply_batch_binary_attacks(
            [base_image],
            action_coords,
            action_values,
            [selected_meta],
            selected_masks,
            config,
        )[0]
        modified_images.append(rebuilt)

    return modified_images


def _predict_seg_maps(model, images, train_data, config, use_adv_predict=False):
    preds = []
    for i, img in enumerate(images):
        if config["dataset"] == "CT_Abd":
            pred = infer_class_map_from_image_and_boxes(model, img, train_data.bboxes[i], train_data.class_ids[i])
        elif use_adv_predict:
            _, pred = adv_model_predict(model, img, config)
        else:
            pred = inference_model(model, img).pred_sem_seg.data.squeeze()

        if not torch.is_tensor(pred):
            pred = torch.from_numpy(np.asarray(pred))
        preds.append(pred.detach().clone())
    return preds


def _build_discrepancy_masks_from_preds(preds, train_data, config):
    masks = []
    device = torch.device(config["device"])
    for i, pred in enumerate(preds):
        pred_t = pred.to(device)
        if config["use_gt"] is True:
            gt = train_data.gt_images[i]
            gt_t = torch.from_numpy(gt).to(device) if not torch.is_tensor(gt) else gt.to(device)
            if config["dataset"] == "ade20k":
                masks.append((pred_t != (gt_t - 1)).int())
            else:
                masks.append((pred_t != gt_t).int())
        else:
            benign = train_data.benign_pred[i]
            if not torch.is_tensor(benign):
                benign = torch.from_numpy(np.asarray(benign))
            masks.append((pred_t != benign.to(device)).int())
    return masks


def attack_iterative_save(model, train_data, config):
    """
    iterative_save True    
    """

    env = Environment.SegEnv(model, config=config)
    length = len(train_data)
    update_images_memory = train_data.images
    env.all_same_shape = (update_images_memory[0].shape == update_images_memory[1].shape)



    if config["show_effect"] == True:
        original_images = copy.deepcopy(train_data.images)
        dis_positive_list = [0]
        dis_negative_list = [0]



    


    # environment 

    if env.all_same_shape == True:
        # update_images shape    
        print(update_images_memory[0].shape)
        B, H, W, C = len(update_images_memory), *update_images_memory[0].shape      # B=N
        # (N,H,W)    
        env.ref_probs   = torch.empty(B, H, W)
        env.ref_classes = torch.empty(B, H, W, dtype=torch.long)
        env.valid_masks  = torch.empty(B, H, W, dtype=torch.long)
        env.action_rgb  = torch.empty(B, 3)  # majority/minority
        env.discrepancy_masks = torch.empty(B, H, W, dtype=torch.long)
        update_masks_memory = torch.empty(B, H, W, dtype=torch.long)
        update_probs_memory = torch.empty(B, H, W)
        train_data.gt_images = torch.stack([torch.from_numpy(img) for img in train_data.gt_images]).to(config["device"])
        train_data.benign_pred = torch.empty(B, H, W).to(config["device"])
        if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
            update_pred_memory = torch.empty(B, H, W, dtype=torch.long)

   


        
    else:

        update_masks_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
        update_probs_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
        if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
            update_pred_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
            

        if train_data.use_gt == False:
            print("Use benign prediction as reference")
            batch_size = 1
            benign_pred = []
            
            for i in range(0, len(train_data.images), batch_size):
                benign_pred.append(torch.zeros(train_data.images[i].shape[:2], device=config["device"]))
                
            train_data.benign_pred = benign_pred
    
    

    #   transform (ex: resize & crop)
    torchvision_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0, 255] → [0.0, 1.0] + (HWC → CHW)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 
                            std=[0.229, 0.224, 0.225])   # ImageNet 
    ])

    batch = config["batch"]
    step = int(length / batch)

    it = 1
    total = []
    box_count = torch.zeros(length)
    iteration = torch.zeros(length).to('cpu')
    # agent = Adversarial_RL_simple.REINFORCE_Policy(config).to(config["device"])
    #  attack_pixels 
    attack_pixels = []
    for img in train_data.images:
        attack = int((img.shape[0] * img.shape[1]) * config["attack_pixel"])
        attack = max(1, attack)
        attack_pixels.append(attack)

    # Forget Process  (bound )
    for p in tqdm(range(config["bound"]), desc="Forget Process"):


        

        if config["backbone"] == "resnet50":
            if config["use_lora"] == True:
                if p == 0:
                    backbone = get_resnet_backbone()
                    loraize_cnn(backbone, r=4, alpha=16, dropout=0, freeze_base=True)
                    agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])

            else:
                backbone = get_resnet_backbone(pretrained=config["pretrained"])
                agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])

        elif config["backbone"] == "conv":

            agent = Adversarial_RL_simple.load_agent(config).to(config["device"])

        
        # kaiming_init_he(agent)
        flag = False
        stop_count = 0
        update_rewards_memory = torch.zeros(length)
        if p == 0:
            max_inner_it = config["max_inner_it"]
            inner_it = 0
            
        else:
            max_inner_it = max(config["max_inner_it"], config["max_inner_it"] + inner_delta)
            inner_it = 0
        
        while True:
            it += 1
            inner_it += 1
            
            change_train_x = []
            total_change_list = torch.tensor([])


            probs_list = []
            discrepancy_mask_list = []
            if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                pred_list = []


                


            for idx in range(step+1):
                start_idx = idx * batch
                end_idx = min((idx + 1) * batch, length)
                if start_idx >= length:
                    break

                imges = train_data.images[start_idx:end_idx]
                gt_list = train_data.gt_images[start_idx:end_idx]

                if config["use_gt"] == False:
                    ref_preds = train_data.benign_pred[start_idx:end_idx]


                # action sampling
                action_means, action_stds = agent(data_preprocessing(imges, torchvision_transform).to(config["device"]))
                if torch.isnan(action_means).any():
                    # NaN    .
                    agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])
                    action_means, action_stds = agent(data_preprocessing(imges, torchvision_transform).to(config["device"]))
                action_stds = torch.clamp(action_stds, 0.1, 10).to('cpu')
                action_means = torch.clamp(action_means, -8, 8).to('cpu')   

                # action processing
                actions_list = []
                actions_logprob_list = []
                batch_attack_pixels = attack_pixels[start_idx:end_idx]

                for i in range(len(imges)):
                    action, logprob = sample_action(action_means[i], action_stds[i], batch_attack_pixels[i])
                    actions_list.append(action)
                    actions_logprob_list.append(logprob.sum().unsqueeze(0))
                actions_logprob = torch.cat(actions_logprob_list)

                # interaction with environment
                if config["use_gt"] == True:
                    if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                        rewards, changed_images, probs_ch, pred_ch = env.step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list)
                        if env.all_same_shape == True:
                            pred_list.append(pred_ch.cpu())
                        else:
                            pred_list.append(pred_ch)
                    else:
                        rewards, changed_images, probs_ch, discrepancy_mask_new = env.step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list)
                        discrepancy_mask_list.append(discrepancy_mask_new.cpu())
                else:
                    if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                        rewards, changed_images, probs_ch, pred_ch = env.step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list, ref_preds)
                        if env.all_same_shape == True:  
                            pred_list.append(pred_ch.cpu())
                        else:
                            pred_list.append(pred_ch)
                    else:
                        rewards, changed_images, probs_ch, discrepancy_mask_new = env.step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list, ref_preds)
                        if env.all_same_shape == True:
                            discrepancy_mask_list.append(discrepancy_mask_new.cpu())
                        else:
                            discrepancy_mask_list.append(discrepancy_mask_new)

                
    
                #    reward 
                if env.all_same_shape == True:
                    probs_list.append(probs_ch.cpu())
                else:
                    probs_list.append(probs_ch)
                
                change_train_x.extend(changed_images)
                total_change_list = torch.cat((total_change_list, rewards.detach().cpu()), dim=0)

                # RL  
                agent.r = rewards
                agent.prob = actions_logprob
                agent.train_net()

            if config["update_valid_action"] == True:
                env.action_mask_init = False

            if env.all_same_shape:
                probs_list = torch.cat(probs_list, dim=0)
                
                if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                    pred_list = torch.cat(pred_list, dim=0)
                else:
                    discrepancy_mask_list = torch.cat(discrepancy_mask_list, dim=0)
            else:
                discrepancy_mask_list = [mask for sublist in discrepancy_mask_list for mask in sublist]
                probs_list = [prob for sublist in probs_list for prob in sublist]
                if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                    pred_list = [pred for sublist in pred_list for pred in sublist]




            standard = update_rewards_memory.mean()
            update_rewards_indices = (total_change_list > update_rewards_memory)
            update_rewards_memory = update(update_rewards_memory, total_change_list, update_rewards_indices, env.all_same_shape)
            update_images_memory = update(update_images_memory, change_train_x, update_rewards_indices, env.all_same_shape)
            update_probs_memory = update(update_probs_memory, probs_list, update_rewards_indices, env.all_same_shape)

            if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                update_pred_memory = update(update_pred_memory, pred_list, update_rewards_indices, env.all_same_shape)
                
            else:
                update_masks_memory = update(update_masks_memory, discrepancy_mask_list, update_rewards_indices, env.all_same_shape)

            iteration[update_rewards_indices == 1] = it
            

            
            epsilon = 1e-8
            # print(f'metric_value: {(update_rewards_memory.mean() - standard) / max(abs(standard), epsilon)}, max_inner_it: {max_inner_it}, inner_it: {inner_it}, max(abs(standard), epsilon): {max(abs(standard), epsilon)}, standard: {standard}')

            stop_count, flag = early_stopping(metric_value=(update_rewards_memory.mean() - standard) / max(abs(standard), epsilon),
                                              patience_counter=stop_count,
                                              min_improvement=config["limit"],
                                              max_patience=config["patient"])
            # new ending condition
            if config["it_max"] is not None:
                if it >= int(config["it_max"]):
                    flag = True

            if flag:
                if config["show_effect"] == True:
                    if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                        dis_positive_count = 0
                        dis_negative_count = 0
                        for i in range(len(update_pred_memory)):
                            if p == 0:
                                dis_t = (train_data.benign_pred[i] != train_data.benign_pred[i]).to(torch.int64)
                                dis_t_new = (train_data.benign_pred[i] != update_pred_memory[i].to(config["device"])).to(torch.int64)
                            else:
                                dis_t = (train_data.benign_pred[i] != previous_pred_memory[i].to(config["device"])).to(torch.int64)
                                dis_t_new = (train_data.benign_pred[i] != update_pred_memory[i].to(config["device"])).to(torch.int64)
                            positive_mask = (dis_t_new - dis_t) > 0
                            negative_mask = (dis_t_new - dis_t) < 0
                            dis_positive_count += positive_mask.sum()
                            dis_negative_count += negative_mask.sum()
                        previous_pred_memory = copy.deepcopy(update_pred_memory)
                        dis_positive_list.append(dis_positive_count/len(update_pred_memory))
                        dis_negative_list.append(dis_negative_count/len(update_pred_memory))
                        print(f'dis_positive_list: {dis_positive_list[-1]}, dis_negative_list: {dis_negative_list[-1]}')


                    elif config["reward_type"] == "discrepancy":
                        dis_positive_count = 0
                        dis_negative_count = 0
                        for i in range(len(update_images_memory)):
                            if p == 0:
                                dis_t = (train_data.benign_pred[i] != train_data.benign_pred[i]).to(torch.int64)
                                dis_t_new = update_masks_memory[i]
                            else:
                                dis_t = previous_masks_memory[i]
                                dis_t_new = update_masks_memory[i]
                            positive_mask = (dis_t_new.to(config["device"]) - dis_t.to(config["device"])) > 0
                            negative_mask = (dis_t_new.to(config["device"]) - dis_t.to(config["device"])) < 0
                            dis_positive_count += positive_mask.sum()
                            dis_negative_count += negative_mask.sum()
                        previous_masks_memory = copy.deepcopy(update_masks_memory)
                        dis_positive_list.append(dis_positive_count/len(update_masks_memory))
                        dis_negative_list.append(dis_negative_count/len(update_masks_memory))
                        print(f'dis_positive_list: {dis_positive_list[-1]}, dis_negative_list: {dis_negative_list[-1]}')

                # env.init = True
                train_data.images = update_images_memory
                env.ref_probs = update_probs_memory


                
                if config["reward_type"] == "standard":
                    env.ref_classes = update_pred_memory
                    
                elif config["reward_type"] == "discrepancy":
                    # if config["update_valid_mask"] == True:
                    #     for i in range(len(env.discrepancy_masks)):
                    #         env.valid_masks[i] = env.valid_masks[i] - ((update_masks_memory[i] - env.discrepancy_masks[i])>0).int()*env.valid_masks[i]
                    env.discrepancy_masks = update_masks_memory

                elif config["reward_type"] == "reduction":
                    if env.all_same_shape == True:
                        env.valid_masks = env.valid_masks - (env.ref_classes != update_pred_memory)*env.valid_masks

                    elif env.all_same_shape == False:
                        for i in range(len(env.valid_masks)):
                            env.valid_masks[i] = env.valid_masks[i] - (env.ref_classes[i] != update_pred_memory[i])*env.valid_masks[i]
                    env.ref_classes = update_pred_memory
                    


                if config["use_lora"] == True:
                    agent.init_lora_and_head()

                inner_delta = max_inner_it - inner_it
                break
            else:
                env.ori_init = False
                env.init = False

        box_count += update_rewards_memory
        total.append(box_count.mean().item())
        print(f'Forget:{p}, changing pixel prediction: {total[-1]}')

        if config["it_max"] is not None:
            if it >= int(config["it_max"]):
                break


        if (p + 1) % (config["bound"]/5) == 0 and p + 1 < config["bound"]:
            if config["show_effect"] == True:
                yield update_images_memory, iteration, dis_positive_list, dis_negative_list
            else:
                yield update_images_memory, iteration


    if config["show_effect"] == True:
        yield update_images_memory, iteration, dis_positive_list, dis_negative_list
    else:
        yield update_images_memory, iteration


def attack_iterative_save_reg(model, train_data, config):
    """
    iterative_save True    
    """

    env = Environment.SegEnv(model, config=config)
    length = len(train_data)
    update_images_memory = train_data.images
    env.all_same_shape = (update_images_memory[0].shape == update_images_memory[1].shape)



    if config["show_effect"] == True:
        original_images = copy.deepcopy(train_data.images)
        dis_positive_list = [0]
        dis_negative_list = [0]


    # environment 



    update_masks_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
    update_pred_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
        

    if train_data.use_gt == False:
        print("Use benign prediction as reference")
        batch_size = 1
        benign_pred = []
        
        for i in range(0, len(train_data.images), batch_size):
            benign_pred.append(torch.zeros(train_data.images[i].shape[:2], device=config["device"]))
            
        train_data.benign_pred = benign_pred

    



    batch = config["batch"]
    step = int(length / batch)

    it = 1
    total = []
    box_count = torch.zeros(length)
    iteration = torch.zeros(length).to('cpu')


    # Forget Process  (bound )
    for p in tqdm(range(config["bound"]), desc="Forget Process"):


        

        if config["backbone"] == "resnet50":
            if config["use_lora"] == True:
                if p == 0:
                    backbone = get_resnet_backbone()
                    loraize_cnn(backbone, r=4, alpha=16, dropout=0, freeze_base=True)
                    agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])

            else:
                backbone = get_resnet_backbone(pretrained=config["pretrained"])
                agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])

        elif config["backbone"] == "conv":

            agent = Adversarial_RL_simple.load_agent(config)

        
        # kaiming_init_he(agent)
        flag = False
        stop_count = 0
        update_rewards_memory = torch.zeros(length)
        if p == 0:
            max_inner_it = config["max_inner_it"]
            inner_it = 0
            
        else:
            max_inner_it = max(config["max_inner_it"], config["max_inner_it"] + inner_delta)
            inner_it = 0
        
        # step1 initial segmentation
        
        if p == 0:
            for idx in range(step+1):
                start_idx = idx * batch
                end_idx = min((idx + 1) * batch, length)
                if start_idx >= length:
                    break

                imges = train_data.images[start_idx:end_idx]
                gt_list = train_data.gt_images[start_idx:end_idx]
                if config["dataset"] == "CT_Abd":
                    bbox_list = train_data.bboxes[start_idx:end_idx]
                    class_ids_list = train_data.class_ids[start_idx:end_idx]


                if config["use_gt"] == False:
                    ref_preds = train_data.benign_pred[start_idx:end_idx]

                if config["dataset"] == "CT_Abd":
                    env._seg_initial_step(imges, gt_list, ref_preds, bbox_list, class_ids_list)
                else: 
                    env._seg_initial_step(imges, gt_list, ref_preds)
                env.init = False

            train_data_attack_set = AttackDataset(train_data.images, train_data.benign_pred, config)

            print(f"✅ Total samples in dataset: {len(train_data_attack_set)}")



            sample_data = train_data_attack_set[0]

            sample_input = sample_data['input']
            sample_mask  = sample_data['mask']
            sample_meta  = sample_data['meta']
            sample_full_mask = sample_data['full_mask']

            print("-" * 30)
            print(f"📊 [Input Shape]: {sample_input.shape}")  # (Batch, 3, 256, 256) 
            print(f"📊 [Mask Shape]:  {sample_mask.shape}")   # (Batch, 1, 256, 256) 
            print(f"📊 [Input Range]: {sample_input.min():.2f} ~ {sample_input.max():.2f}")
            print(f"📊 [Full Mask Shape]: {sample_full_mask.shape}")
            print("-" * 30)

            # 3. (BBox ) 
            print(f"🏷️  [Metadata Sample]:")
            for i, meta in enumerate(sample_meta):
                print(f"   - Class {meta['class_id']}: BBox {meta['bbox']}")

        class_level_update = (config.get("reg_update_level", "image") == "class")
        # print(f"class_level_update: {class_level_update}")
        if class_level_update:
            update_class_rewards_memory = _init_class_reward_memory(train_data_attack_set)
            update_class_perturb_memory = _init_class_perturb_memory(train_data_attack_set)
            base_images_for_class_update = [img.copy() for img in train_data.images]


            
            


        # step2 attack start
        while True:
            it += 1
            inner_it += 1
            change_train_x = []
            total_change_list = torch.tensor([])
            if class_level_update:
                standard = update_rewards_memory.mean()
                update_rewards_indices = torch.zeros(length, dtype=torch.bool)
            discrepancy_mask_list = []
            
            pred_list = []


            for idx in range(step+1):
                start_idx = idx * batch
                end_idx = min((idx + 1) * batch, length)
                if start_idx >= length:
                    break

                imges = train_data.images[start_idx:end_idx]
                gt_list = train_data.gt_images[start_idx:end_idx]
                batch_attack_samples = train_data_attack_set[start_idx:end_idx]
                if config["dataset"] == "CT_Abd":
                    bboxes_list = train_data.bboxes[start_idx:end_idx]
                    class_ids_list = train_data.class_ids[start_idx:end_idx]
                

                if config["use_gt"] == False:
                    ref_preds = train_data.benign_pred[start_idx:end_idx]

                


                # action sampling
                masks = torch.cat([sample['mask'] for sample in batch_attack_samples], dim=0).to(config["device"])
                regions = torch.cat([sample['input'] for sample in batch_attack_samples], dim=0).to(config["device"])
                batch_attack_mask_set = [sample['full_mask'] for sample in batch_attack_samples]
                meta_list = [sample['meta'] for sample in batch_attack_samples]
                xy_coords, rgb_actions, total_log_prob = agent(regions ,masks, meta_list)


                # action processing
                actions_list = []

                
                # make transformed images
                changed_images = apply_batch_binary_attacks(imges, xy_coords, rgb_actions, meta_list, masks,config)



                # interaction with environment
                if config["use_gt"] == True:
                    if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                        rewards, changed_images, probs_ch, pred_ch = env.reg_step(changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list)
                        if env.all_same_shape == True:
                            pred_list.append(pred_ch.cpu())
                        else:
                            pred_list.append(pred_ch)
                    else:
                        rewards, changed_images, probs_ch, discrepancy_mask_new = env.reg_step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list)
                        discrepancy_mask_list.append(discrepancy_mask_new.cpu())
                else:
                    if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                        if config["dataset"] == "CT_Abd":
                            rewards, changed_images, pred_ch, memory_reward = env.reg_step(changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list)
                        else:
                            rewards, changed_images, pred_ch, memory_reward = env.reg_step(changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list, bboxes_list=bboxes_list, class_ids_list = class_ids_list)
                        pred_list.append(pred_ch)
                    else:
                        if config["dataset"] == "CT_Abd":
                            rewards, changed_images, pred_ch, discrepancy_mask_new, memory_reward = env.reg_step(changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list, ref_preds, bboxes_list=bboxes_list, class_ids_list = class_ids_list)
                        else:
                            rewards, changed_images, pred_ch, discrepancy_mask_new, memory_reward = env.reg_step(changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list, ref_preds)
                        pred_list.append(pred_ch)
                        discrepancy_mask_list.append(discrepancy_mask_new)



                # memory reward calculation
                total_change_list = torch.cat((total_change_list, memory_reward.detach().cpu()), dim=0)
                if class_level_update:
                    aligned_actions = _align_actions_with_meta(xy_coords, rgb_actions, meta_list)
                    _update_class_memories_from_batch(
                        rewards.detach().cpu(),
                        start_idx,
                        meta_list,
                        aligned_actions,
                        update_class_rewards_memory,
                        update_class_perturb_memory,
                        update_rewards_indices,
                        update_rewards_memory,
                    )


                
                change_train_x.extend(changed_images)


                # RL  
                agent.r = rewards
                agent.prob = total_log_prob
                agent.train_net()

            if config["update_valid_action"] == True:
                env.action_mask_init = False


            if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                pred_list = [pred for sublist in pred_list for pred in sublist]

            else:
                pred_list = [pred for sublist in pred_list for pred in sublist]
                discrepancy_mask_list = [mask for sublist in discrepancy_mask_list for mask in sublist]




            if class_level_update is False:
                standard = update_rewards_memory.mean()
                update_rewards_indices = (total_change_list > update_rewards_memory)
                update_rewards_memory = update(update_rewards_memory, total_change_list, update_rewards_indices, env.all_same_shape)
            update_images_memory = update(update_images_memory, change_train_x, update_rewards_indices, env.all_same_shape)


            if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                update_pred_memory = update(update_pred_memory, pred_list, update_rewards_indices, env.all_same_shape)
                print(f"iteration: {it}, update_pred_memory: {len(update_pred_memory)}, pred_list: {len(pred_list)}")
                
            else:
                update_pred_memory = update(update_pred_memory, pred_list, update_rewards_indices, env.all_same_shape)
                update_masks_memory = update(update_masks_memory, discrepancy_mask_list, update_rewards_indices, env.all_same_shape)

            iteration[update_rewards_indices == 1] = it
            

            
            epsilon = 1e-8
            print(f'metric_value: {(update_rewards_memory.mean() - standard) / max(abs(standard), epsilon)}, max_inner_it: {max_inner_it}, inner_it: {inner_it}, max(abs(standard), epsilon): {max(abs(standard), epsilon)}, standard: {standard}')

            stop_count, flag = early_stopping(metric_value=(update_rewards_memory.mean() - standard) / max(abs(standard), epsilon),
                                              patience_counter=stop_count,
                                              min_improvement=config["limit"],
                                              max_patience=config["patient"])
            # new ending condition
            if config["it_max"] is not None:
                if it >= int(config["it_max"]):
                    flag = True

            if flag:
                if class_level_update:
                    update_images_memory = _compose_images_from_class_perturb_memory(
                        base_images_for_class_update,
                        train_data_attack_set,
                        update_class_perturb_memory,
                        config,
                    )
                    it+=1
                    update_pred_memory = _predict_seg_maps(
                        model,
                        update_images_memory,
                        train_data,
                        config,
                        use_adv_predict=False,
                    )
                    if config["reward_type"] == "discrepancy":
                        update_masks_memory = _build_discrepancy_masks_from_preds(
                            update_pred_memory, train_data, config
                        )

                if config["show_effect"] == True:
                    if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                        dis_positive_count = 0
                        dis_negative_count = 0
                        for i in range(len(update_pred_memory)):
                            if p == 0:
                                dis_t = (train_data.benign_pred[i] != train_data.benign_pred[i]).to(torch.int64)
                                dis_t_new = (train_data.benign_pred[i] != update_pred_memory[i].to(config["device"])).to(torch.int64)
                            else:
                                dis_t = (train_data.benign_pred[i] != previous_pred_memory[i].to(config["device"])).to(torch.int64)
                                dis_t_new = (train_data.benign_pred[i] != update_pred_memory[i].to(config["device"])).to(torch.int64)
                            positive_mask = (dis_t_new - dis_t) > 0
                            negative_mask = (dis_t_new - dis_t) < 0
                            dis_positive_count += positive_mask.sum()
                            dis_negative_count += negative_mask.sum()
                        previous_pred_memory = copy.deepcopy(update_pred_memory)
                        dis_positive_list.append(dis_positive_count/len(update_pred_memory))
                        dis_negative_list.append(dis_negative_count/len(update_pred_memory))
                        print(f'dis_positive_list: {dis_positive_list[-1]}, dis_negative_list: {dis_negative_list[-1]}')


                    elif config["reward_type"] == "discrepancy":
                        dis_positive_count = 0
                        dis_negative_count = 0
                        for i in range(len(update_images_memory)):
                            if p == 0:
                                dis_t = (train_data.benign_pred[i] != train_data.benign_pred[i]).to(torch.int64)
                                dis_t_new = update_masks_memory[i]
                            else:
                                dis_t = previous_masks_memory[i]
                                dis_t_new = update_masks_memory[i]
                            positive_mask = (dis_t_new.to(config["device"]) - dis_t.to(config["device"])) > 0
                            negative_mask = (dis_t_new.to(config["device"]) - dis_t.to(config["device"])) < 0
                            dis_positive_count += positive_mask.sum()
                            dis_negative_count += negative_mask.sum()
                        previous_masks_memory = copy.deepcopy(update_masks_memory)
                        dis_positive_list.append(dis_positive_count/len(update_masks_memory))
                        dis_negative_list.append(dis_negative_count/len(update_masks_memory))
                        print(f'dis_positive_list: {dis_positive_list[-1]}, dis_negative_list: {dis_negative_list[-1]}')

                # env.init = True
                train_data.images = update_images_memory
                if config["delete_attack_pixels"] == True:
                    for i in range(len(update_pred_memory)):
                        train_data_attack_set.delete_attack_pixels(i, update_pred_memory[i])


                
                if config["reward_type"] == "standard":
                    env.ref_classes = update_pred_memory
                    
                elif config["reward_type"] == "discrepancy":
                    # if config["update_valid_mask"] == True:
                    #     for i in range(len(env.discrepancy_masks)):
                    #         env.valid_masks[i] = env.valid_masks[i] - ((update_masks_memory[i] - env.discrepancy_masks[i])>0).int()*env.valid_masks[i]
                    env.discrepancy_masks = update_masks_memory

                elif config["reward_type"] == "reduction":
                    if env.all_same_shape == True:
                        env.valid_masks = env.valid_masks - (env.ref_classes != update_pred_memory)*env.valid_masks

                    elif env.all_same_shape == False:
                        for i in range(len(env.valid_masks)):
                            env.valid_masks[i] = env.valid_masks[i] - (env.ref_classes[i] != update_pred_memory[i])*env.valid_masks[i]
                    env.ref_classes = update_pred_memory

                    


                if config["use_lora"] == True:
                    agent.init_lora_and_head()

                inner_delta = max_inner_it - inner_it
                break
            else:
                env.ori_init = False
                env.init = False

        box_count += update_rewards_memory
        total.append(box_count.mean().item())
        print(f'Forget:{p}, changing pixel prediction: {total[-1]}')

        if config["it_max"] is not None:
            if it >= int(config["it_max"]):
                break


        if (p + 1) % (config["bound"]/5) == 0 and p + 1 < config["bound"]:
            if config["show_effect"] == True:
                yield update_images_memory, iteration, dis_positive_list, dis_negative_list
            else:
                yield update_images_memory, iteration


    if config["show_effect"] == True:
        yield update_images_memory, iteration, dis_positive_list, dis_negative_list
    else:
        yield update_images_memory, iteration

def adv_setting_attack_iterative_save_reg(model, train_data, config):
    """
    adv_setting (reg action space) iterative_save .
    """
    env = Environment.SegEnv(model, config=config)
    length = len(train_data)
    update_images_memory = train_data.images
    env.all_same_shape = (update_images_memory[0].shape == update_images_memory[1].shape)

    if config["show_effect"] == True:
        dis_positive_list = [0]
        dis_negative_list = [0]

    update_masks_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
    update_pred_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]

    if train_data.use_gt is False:
        print("Use benign prediction as reference")
        benign_pred = []
        for i in range(len(train_data.images)):
            benign_pred.append(torch.zeros(train_data.images[i].shape[:2], device=config["device"], dtype=torch.long))
        train_data.benign_pred = benign_pred

    batch = config["batch"]
    step = int(length / batch)

    it = 1
    total = []
    box_count = torch.zeros(length)
    iteration = torch.zeros(length).to("cpu")

    for p in tqdm(range(config["bound"]), desc="Forget Process"):
        if config["backbone"] == "resnet50":
            if config["use_lora"] is True:
                if p == 0:
                    backbone = get_resnet_backbone()
                    loraize_cnn(backbone, r=4, alpha=16, dropout=0, freeze_base=True)
                    agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])
            else:
                backbone = get_resnet_backbone(pretrained=config["pretrained"])
                agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])
        elif config["backbone"] == "conv":
            agent = Adversarial_RL_simple.load_agent(config)

        flag = False
        stop_count = 0
        update_rewards_memory = torch.zeros(length)
        if p == 0:
            max_inner_it = config["max_inner_it"]
            inner_it = 0
        else:
            max_inner_it = max(config["max_inner_it"], config["max_inner_it"] + inner_delta)
            inner_it = 0

        if p == 0:
            for idx in range(step + 1):
                start_idx = idx * batch
                end_idx = min((idx + 1) * batch, length)
                if start_idx >= length:
                    break

                imges = train_data.images[start_idx:end_idx]
                gt_list = train_data.gt_images[start_idx:end_idx]
                ref_preds = train_data.benign_pred[start_idx:end_idx] if config["use_gt"] is False else None
                env._adv_reg_initial_step(imges, gt_list, ref_preds)
                env.init = False

            train_data_attack_set = AttackDataset(train_data.images, train_data.benign_pred, config)

        class_level_update = (config.get("reg_update_level", "image") == "class")
        if class_level_update:
            update_class_rewards_memory = _init_class_reward_memory(train_data_attack_set)
            update_class_perturb_memory = _init_class_perturb_memory(train_data_attack_set)
            base_images_for_class_update = [img.copy() for img in train_data.images]

        while True:
            it += 1
            inner_it += 1
            change_train_x = []
            total_change_list = torch.tensor([])
            if class_level_update:
                standard = update_rewards_memory.mean()
                update_rewards_indices = torch.zeros(length, dtype=torch.bool)
            discrepancy_mask_list = []
            pred_list = []

            for idx in range(step + 1):
                start_idx = idx * batch
                end_idx = min((idx + 1) * batch, length)
                if start_idx >= length:
                    break

                imges = train_data.images[start_idx:end_idx]
                batch_attack_samples = train_data_attack_set[start_idx:end_idx]

                masks = torch.cat([sample["mask"] for sample in batch_attack_samples], dim=0).to(config["device"])
                regions = torch.cat([sample["input"] for sample in batch_attack_samples], dim=0).to(config["device"])
                batch_attack_mask_set = [sample["full_mask"] for sample in batch_attack_samples]
                meta_list = [sample["meta"] for sample in batch_attack_samples]
                xy_coords, rgb_actions, total_log_prob = agent(regions, masks, meta_list)

                changed_images = apply_batch_binary_attacks(imges, xy_coords, rgb_actions, meta_list, masks, config)

                if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                    rewards, changed_images, pred_ch, memory_reward = env._adv_reg_step(
                        changed_images, start_idx, batch_attack_mask_set, meta_list
                    )
                    pred_list.append(pred_ch)
                else:
                    rewards, changed_images, pred_ch, discrepancy_mask_new, memory_reward = env._adv_reg_step(
                        changed_images, start_idx, batch_attack_mask_set, meta_list
                    )
                    pred_list.append(pred_ch)
                    discrepancy_mask_list.append(discrepancy_mask_new)

                total_change_list = torch.cat((total_change_list, memory_reward.detach().cpu()), dim=0)
                if class_level_update:
                    aligned_actions = _align_actions_with_meta(xy_coords, rgb_actions, meta_list)
                    _update_class_memories_from_batch(
                        rewards.detach().cpu(),
                        start_idx,
                        meta_list,
                        aligned_actions,
                        update_class_rewards_memory,
                        update_class_perturb_memory,
                        update_rewards_indices,
                        update_rewards_memory,
                    )
                change_train_x.extend(changed_images)

                agent.r = rewards
                agent.prob = total_log_prob
                agent.train_net()

            if config["update_valid_action"] is True:
                env.action_mask_init = False

            if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                pred_list = [pred for sublist in pred_list for pred in sublist]
            else:
                pred_list = [pred for sublist in pred_list for pred in sublist]
                discrepancy_mask_list = [mask for sublist in discrepancy_mask_list for mask in sublist]

            if class_level_update is False:
                standard = update_rewards_memory.mean()
                update_rewards_indices = (total_change_list > update_rewards_memory)
                update_rewards_memory = update(update_rewards_memory, total_change_list, update_rewards_indices, env.all_same_shape)
            update_images_memory = update(update_images_memory, change_train_x, update_rewards_indices, env.all_same_shape)

            if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                update_pred_memory = update(update_pred_memory, pred_list, update_rewards_indices, env.all_same_shape)
            else:
                update_pred_memory = update(update_pred_memory, pred_list, update_rewards_indices, env.all_same_shape)
                update_masks_memory = update(update_masks_memory, discrepancy_mask_list, update_rewards_indices, env.all_same_shape)

            iteration[update_rewards_indices == 1] = it

            epsilon = 1e-8
            stop_count, flag = early_stopping(
                metric_value=(update_rewards_memory.mean() - standard) / max(abs(standard), epsilon),
                patience_counter=stop_count,
                min_improvement=config["limit"],
                max_patience=config["patient"],
            )
            if config["it_max"] is not None and it >= int(config["it_max"]):
                flag = True

            if flag:
                if class_level_update:
                    update_images_memory = _compose_images_from_class_perturb_memory(
                        base_images_for_class_update,
                        train_data_attack_set,
                        update_class_perturb_memory,
                        config,
                    )
                    it+=1
                    update_pred_memory = _predict_seg_maps(
                        model,
                        update_images_memory,
                        train_data,
                        config,
                        use_adv_predict=True,
                    )
                    if config["reward_type"] == "discrepancy":
                        update_masks_memory = _build_discrepancy_masks_from_preds(
                            update_pred_memory, train_data, config
                        )

                train_data.images = update_images_memory
                if config["delete_attack_pixels"] is True:
                    for i in range(len(update_pred_memory)):
                        train_data_attack_set.delete_attack_pixels(i, update_pred_memory[i])

                if config["reward_type"] == "standard":
                    env.ref_classes = update_pred_memory
                elif config["reward_type"] == "discrepancy":
                    env.discrepancy_masks = update_masks_memory
                elif config["reward_type"] == "reduction":
                    for i in range(len(env.valid_masks)):
                        matches = (env.ref_classes[i] == update_pred_memory[i])
                        env.valid_masks[i] *= matches.to(env.valid_masks[i].dtype)
                    env.ref_classes = update_pred_memory

                if config["use_lora"] is True:
                    agent.init_lora_and_head()

                inner_delta = max_inner_it - inner_it
                break
            else:
                env.ori_init = False
                env.init = False

        box_count += update_rewards_memory
        total.append(box_count.mean().item())
        print(f"Forget:{p}, changing pixel prediction: {total[-1]}")

        if config["it_max"] is not None and it >= int(config["it_max"]):
            break

        if ((p + 1) % (config["bound"] / 5) == 0 and p + 1 < config["bound"]):
            if config["show_effect"] == True:
                yield update_images_memory, iteration, dis_positive_list, dis_negative_list
            else:
                yield update_images_memory, iteration

    if config["show_effect"] == True:
        yield update_images_memory, iteration, dis_positive_list, dis_negative_list
    else:
        yield update_images_memory, iteration


def adv_setting_attack_iterative_save(model, train_data, config):
    """
    iterative_save True    
    """

    env = Environment.SegEnv(model, config=config)
    length = len(train_data)
    update_images_memory = train_data.images
    env.all_same_shape = (update_images_memory[0].shape == update_images_memory[1].shape)

    if config["show_effect"] == True:
        dis_positive_list = [0]
        dis_negative_list = [0]



    #  gt    gt 
    

    # environment 

    if env.all_same_shape == True:
        # update_images shape    
        print(update_images_memory[0].shape)
        B, H, W, C = len(update_images_memory), *update_images_memory[0].shape      # B=N
        # (N,H,W)    
        env.ref_probs   = torch.empty(B, H, W)
        env.ref_classes = torch.empty(B, H, W, dtype=torch.long)
        env.valid_masks  = torch.empty(B, H, W, dtype=torch.long)
        env.action_rgb  = torch.empty(B, 3)  # majority/minority
        env.discrepancy_masks = torch.empty(B, H, W, dtype=torch.long)
        update_masks_memory = torch.empty(B, H, W, dtype=torch.long)
        update_probs_memory = torch.empty(B, H, W)
        train_data.gt_images = torch.stack([torch.from_numpy(img) for img in train_data.gt_images]).to(config["device"])
        train_data.benign_pred = torch.empty(B, H, W).to(config["device"])
        if config["reward_type"] == "standard":
            update_pred_memory = torch.empty(B, H, W, dtype=torch.long)

        
    else:

        update_masks_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
        update_probs_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]
        if config["reward_type"] == "standard":
            update_pred_memory = [torch.zeros(img.shape[:2], device=config["device"]) for img in update_images_memory]

        if train_data.use_gt == False:
            print("Use benign prediction as reference")
            batch_size = 1
            benign_pred = []
            
            for i in range(0, len(train_data.images), batch_size):
                benign_pred.append(torch.zeros(train_data.images[i].shape[:2], device=config["device"]))
                
            train_data.benign_pred = benign_pred
    
    

    #   transform (ex: resize & crop)
    torchvision_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0, 255] → [0.0, 1.0] + (HWC → CHW)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 
                            std=[0.229, 0.224, 0.225])   # ImageNet 
    ])

    batch = config["batch"]
    step = int(length / batch)

    it = 1
    total = []
    box_count = torch.zeros(length)
    iteration = torch.zeros(length).to('cpu')
    #  attack_pixels 
    attack_pixels = []
    for img in train_data.images:
        attack = int((img.shape[0] * img.shape[1]) * config["attack_pixel"])
        attack = max(1, attack)
        attack_pixels.append(attack)

    # Forget Process  (bound )
    for p in tqdm(range(config["bound"]), desc="Forget Process"):

        

        if config["backbone"] == "resnet50":
            if config["use_lora"] == True:
                if p == 0:
                    backbone = get_resnet_backbone()
                    loraize_cnn(backbone, r=4, alpha=16, dropout=0, freeze_base=True)
                    agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])

            else:
                backbone = get_resnet_backbone(pretrained=config["pretrained"])
                agent = Adversarial_RL_simple.ResNetLoRA_RL(config, backbone).to(config["device"])
        elif config["backbone"] == "conv":
            agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])

        
        # kaiming_init_he(agent)
        flag = False
        stop_count = 0
        update_rewards_memory = torch.zeros(length)
        if p == 0:
            max_inner_it = config["max_inner_it"]
            inner_it = 0
            
        else:
            max_inner_it = max(config["max_inner_it"], config["max_inner_it"] + inner_delta)
            inner_it = 0
        
        while True:
            it += 1
            inner_it += 1
            
            change_train_x = []
            total_change_list = torch.tensor([])


            probs_list = []
            discrepancy_mask_list = []
            if config["reward_type"] == "standard":
                pred_list = []


                


            for idx in range(step+1):
                start_idx = idx * batch
                end_idx = min((idx + 1) * batch, length)
                if start_idx >= length:
                    break

                imges = train_data.images[start_idx:end_idx]
                gt_list = train_data.gt_images[start_idx:end_idx]

                if config["use_gt"] == False:
                    ref_preds = train_data.benign_pred[start_idx:end_idx]


                # action sampling
                action_means, action_stds = agent(data_preprocessing(imges, torchvision_transform).to(config["device"]))
                if torch.isnan(action_means).any():
                    # NaN    .
                    agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])
                    action_means, action_stds = agent(data_preprocessing(imges, torchvision_transform).to(config["device"]))
                action_stds = torch.clamp(action_stds, 0.1, 10).to('cpu')
                action_means = torch.clamp(action_means, -8, 8).to('cpu')   

                # action processing
                actions_list = []
                actions_logprob_list = []
                sampled_logprobs = []
                batch_attack_pixels = attack_pixels[start_idx:end_idx]

                for i in range(len(imges)):
                    action, logprob = sample_action(action_means[i], action_stds[i], batch_attack_pixels[i])
                    actions_list.append(action)
                    sampled_logprobs.append(logprob)

                # interaction with environment
                if config["use_gt"] == True:
                    if config["reward_type"] == "standard":
                        rewards, changed_images, probs_ch, pred_ch = env.adv_setting_step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list)
                        if env.all_same_shape == True:
                            pred_list.append(pred_ch.cpu())
                        else:
                            pred_list.append(pred_ch)
                    else:
                        rewards, changed_images, probs_ch, discrepancy_mask_new = env.adv_setting_step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list)
                        discrepancy_mask_list.append(discrepancy_mask_new.cpu())
                else:
                    if config["reward_type"] == "standard":
                        
                        rewards, changed_images, probs_ch, pred_ch = env.adv_setting_step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list, ref_preds)
                        if env.all_same_shape == True:  
                            pred_list.append(pred_ch.cpu())
                        else:
                            pred_list.append(pred_ch)

                    elif config["reward_type"] == "reduction":

                        rewards, changed_images, probs_ch, pred_ch, valid_idx_list = env.adv_setting_step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list, ref_preds)
                        if env.all_same_shape == True:  
                            pred_list.append(pred_ch.cpu())
                        else:
                            pred_list.append(pred_ch)
                    
                    elif config["reward_type"] == "discrepancy":
                        rewards, changed_images, probs_ch, discrepancy_mask_new = env.adv_setting_step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt_list, ref_preds)
                        if env.all_same_shape == True:
                            discrepancy_mask_list.append(discrepancy_mask_new.cpu())
                        else:
                            discrepancy_mask_list.append(discrepancy_mask_new)


                if config["reward_type"] == "reduction":
                    for i in range(len(imges)):
                        current_logprob = sampled_logprobs[i]
                        if current_logprob.dim() == 0:
                            actions_logprob_list.append(current_logprob.unsqueeze(0))
                        else:
                            actions_logprob_list.append(current_logprob[valid_idx_list[i]].sum().unsqueeze(0))
                else:
                    for i in range(len(imges)):
                        current_logprob = sampled_logprobs[i]
                        if current_logprob.dim() == 0:
                            actions_logprob_list.append(current_logprob.unsqueeze(0))
                        else:
                            actions_logprob_list.append(current_logprob.sum().unsqueeze(0))

                actions_logprob = torch.cat(actions_logprob_list)


                #    reward 
                if env.all_same_shape == True:
                    probs_list.append(probs_ch.cpu())
                else:
                    probs_list.append(probs_ch)
                
                change_train_x.extend(changed_images)
                total_change_list = torch.cat((total_change_list, rewards.detach().cpu()), dim=0)

                # RL  
                agent.r = rewards
                agent.prob = actions_logprob
                agent.train_net()


            if env.all_same_shape:
                probs_list = torch.cat(probs_list, dim=0)
                
                if config["reward_type"] == "standard":
                    pred_list = torch.cat(pred_list, dim=0)
                else:
                    discrepancy_mask_list = torch.cat(discrepancy_mask_list, dim=0)
            else:
                discrepancy_mask_list = [mask for sublist in discrepancy_mask_list for mask in sublist]
                probs_list = [prob for sublist in probs_list for prob in sublist]
                if config["reward_type"] == "standard":
                    pred_list = [pred for sublist in pred_list for pred in sublist]




            standard = update_rewards_memory.mean()
            update_rewards_indices = (total_change_list > update_rewards_memory)
            update_rewards_memory = update(update_rewards_memory, total_change_list, update_rewards_indices, env.all_same_shape)
            update_images_memory = update(update_images_memory, change_train_x, update_rewards_indices, env.all_same_shape)
            update_probs_memory = update(update_probs_memory, probs_list, update_rewards_indices, env.all_same_shape)
            if config["reward_type"] == "standard":
                update_pred_memory = update(update_pred_memory, pred_list, update_rewards_indices, env.all_same_shape)
            else:
                update_masks_memory = update(update_masks_memory, discrepancy_mask_list, update_rewards_indices, env.all_same_shape)

            iteration[update_rewards_indices == 1] = it
            

            
            epsilon = 1e-8
            print(f'metric_value: {(update_rewards_memory.mean() - standard) / max(abs(standard), epsilon)}, max_inner_it: {max_inner_it}, inner_it: {inner_it}, max(abs(standard), epsilon): {max(abs(standard), epsilon)}, standard: {standard}')

            stop_count, flag = early_stopping(metric_value=(update_rewards_memory.mean() - standard) / max(abs(standard), epsilon),
                                              patience_counter=stop_count,
                                              min_improvement=config["limit"],
                                              max_patience=config["patient"])
            # new ending condition
            if inner_it == max_inner_it or inner_it == 2*config["max_inner_it"]:
                flag = True

            if flag:
                if config["show_effect"] == True:
                    if config["reward_type"] == "standard" or config["reward_type"] == "reduction":
                        dis_positive_count = 0
                        dis_negative_count = 0
                        for i in range(len(update_pred_memory)):
                            if p == 0:
                                dis_t = (train_data.benign_pred[i] != train_data.benign_pred[i]).to(torch.int64)
                                dis_t_new = (train_data.benign_pred[i] != update_pred_memory[i].to(config["device"])).to(torch.int64)
                            else:
                                dis_t = (train_data.benign_pred[i] != previous_pred_memory[i].to(config["device"])).to(torch.int64)
                                dis_t_new = (train_data.benign_pred[i] != update_pred_memory[i].to(config["device"])).to(torch.int64)
                            positive_mask = (dis_t_new - dis_t) > 0
                            negative_mask = (dis_t_new - dis_t) < 0
                            dis_positive_count += positive_mask.sum()
                            dis_negative_count += negative_mask.sum()
                        previous_pred_memory = copy.deepcopy(update_pred_memory)
                        dis_positive_list.append(dis_positive_count/len(update_pred_memory))
                        dis_negative_list.append(dis_negative_count/len(update_pred_memory))
                        print(f'dis_positive_list: {dis_positive_list[-1]}, dis_negative_list: {dis_negative_list[-1]}')

                    elif config["reward_type"] == "discrepancy":
                        dis_positive_count = 0
                        dis_negative_count = 0
                        for i in range(len(update_images_memory)):
                            if p == 0:
                                dis_t = (train_data.benign_pred[i] != train_data.benign_pred[i]).to(torch.int64)
                                dis_t_new = update_masks_memory[i]
                            else:
                                dis_t = previous_masks_memory[i]
                                dis_t_new = update_masks_memory[i]
                            positive_mask = (dis_t_new.to(config["device"]) - dis_t.to(config["device"])) > 0
                            negative_mask = (dis_t_new.to(config["device"]) - dis_t.to(config["device"])) < 0
                            dis_positive_count += positive_mask.sum()
                            dis_negative_count += negative_mask.sum()
                        previous_masks_memory = copy.deepcopy(update_masks_memory)
                        dis_positive_list.append(dis_positive_count/len(update_masks_memory))
                        dis_negative_list.append(dis_negative_count/len(update_masks_memory))
                        print(f'dis_positive_list: {dis_positive_list[-1]}, dis_negative_list: {dis_negative_list[-1]}')

                # env.init = True
                train_data.images = update_images_memory
                env.ref_probs = update_probs_memory

                
                if config["reward_type"] == "standard":
                    env.ref_classes = update_pred_memory
                    
                elif config["reward_type"] == "discrepancy":
                    # if config["update_valid_mask"] == True:
                    #     for i in range(len(env.discrepancy_masks)):
                    #         env.valid_masks[i] = env.valid_masks[i] - ((update_masks_memory[i] - env.discrepancy_masks[i])>0).int()*env.valid_masks[i]
                    env.discrepancy_masks = update_masks_memory

                elif config["reward_type"] == "reduction":
                    if env.all_same_shape == True:
                        env.valid_masks = env.valid_masks - (env.ref_classes != update_pred_memory)*env.valid_masks

                    elif env.all_same_shape == False:
                        for i in range(len(env.valid_masks)):
                            env.valid_masks[i] = env.valid_masks[i] - (env.ref_classes[i] != update_pred_memory[i])*env.valid_masks[i]
                    env.ref_classes = update_pred_memory
                    
                    


                if config["use_lora"] == True:
                    agent.init_lora_and_head()

                inner_delta = max_inner_it - inner_it
                break
            else:
                env.ori_init = False
                env.init = False

        box_count += update_rewards_memory
        total.append(box_count.mean().item())
        print(f"Forget:{p}, changing pixel prediction: {total[-1]}")

        if config["it_max"] is not None and it >= int(config["it_max"]):
            break

        if ((p + 1) % (config["bound"] / 5) == 0 and p + 1 < config["bound"]):
            if config["show_effect"] == True:
                yield update_images_memory, iteration, dis_positive_list, dis_negative_list
            else:
                yield update_images_memory, iteration

    if config["show_effect"] == True:
        yield update_images_memory, iteration, dis_positive_list, dis_negative_list
    else:
        yield update_images_memory, iteration



# def attack(model, train_data, config):

#       (train_data)  adversarial  .

#     #  attack     (yield )
#     env = Environment.SegEnv(model, config=config)
#     length = len(train_data)
#     update_images = train_data.images
#     env.all_same_shape = (update_images[0].shape == update_images[1].shape)

#     #   transform (ex: resize & crop)
#     torchvision_transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),  # [0, 255] → [0.0, 1.0] + (HWC → CHW)
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 
#                             std=[0.229, 0.224, 0.225])   # ImageNet 


#     batch = config["batch"]
#     step = int(length / batch)

#     it = 0
    
#     total = []
#     box_count = np.zeros(length)
#     iteration = torch.zeros(length).to('cpu')

#     #  attack_pixels 
#     attack_pixels = []
#     for img in train_data.images:
#         attack = int((img.shape[0] * img.shape[1]) * config["attack_pixel"])
#         attack = max(1, attack)  # 0  1 
#         attack_pixels.append(attack)

#     # Forget Process  (bound )
#     for p in tqdm(range(config["bound"]), desc="Forget Process"):

#         agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])
#         flag = False
#         stop_count = 0
#         update_rewards = np.zeros(length, dtype=np.float32)

#         while True:
#             it += 1
#             change_train_x = []
#             total_change_list = []
#             gt = None

#             for idx in range(step+1):
#                 start_idx = idx * batch
#                 end_idx = min((idx + 1) * batch, length)
#                 if start_idx >= length:
#                     break

#                 imges = train_data.images[start_idx:end_idx]

#                 if env.init == True:
#                     gt = train_data.gt_images[start_idx:end_idx]

#                 # action sampling
#                 action_means, action_stds = agent(data_preprocessing(imges, torchvision_transform).to(config["device"]))
#                 if torch.isnan(action_means).any():
#                     # NaN    .
#                     agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])
#                     action_means, action_stds = agent(data_preprocessing(imges, torchvision_transform).to(config["device"]))
#                 action_stds = torch.clamp(action_stds, 0.1, 10).to('cpu')
#                 action_means = torch.clamp(action_means, -8, 8).to('cpu')   

#                 # action processing
#                 actions_list = []
#                 actions_logprob_list = []
#                 batch_attack_pixels = attack_pixels[start_idx:end_idx]

#                 for i in range(len(imges)):
#                     action, logprob = sample_action(action_means[i], action_stds[i], batch_attack_pixels[i])
#                     actions_list.append(action)
#                     actions_logprob_list.append(logprob.sum().unsqueeze(0))

#                 actions_logprob = torch.cat(actions_logprob_list)

#                 # interaction with environment
#                 rewards, changed_images = env.step(imges, actions_list, start_idx, end_idx, batch_attack_pixels, gt)

#                 #    reward 
#                 change_train_x.extend(changed_images)
#                 total_change_list = np.concatenate((total_change_list, rewards.detach().cpu().numpy()), axis=0)

#                 # RL  
#                 agent.r = rewards
#                 agent.prob = actions_logprob
#                 agent.train_net()


#             standard = update_rewards.mean()
#             epsilon = 1e-8
#             update_rewards_indices = (total_change_list > update_rewards).astype(int)
#             update_rewards = update(update_rewards, total_change_list, update_rewards_indices, env.all_same_shape)
#             update_images = update(update_images, change_train_x, update_rewards_indices, env.all_same_shape)
#             iteration[update_rewards_indices == 1] = it
            

#             stop_count, flag = early_stopping(metric_value=(update_rewards.mean() - standard) / max(abs(standard), epsilon),
#                                                 patience_counter=stop_count,
#                                                 min_improvement=config["limit"],
#                                                 max_patience=config["patient"])

#             if flag:
#                 env.init = True
#                 train_data = CitySet(images=update_images, gt_images=train_data.gt_images)
#                 env.ref_probs = []
#                 env.ref_classes = []    
                

#                 # fix part
#                 env.valid_mask = []

#                 it += 1
#                 break
#             else:
#                 env.ori_init = False
#                 env.init = False


#         box_count += update_rewards
#         total.append(box_count.mean().item())
#         print(f'Forget:{p}, changing pixel prediction: {total[-1]}')



#     return update_images, iteration
