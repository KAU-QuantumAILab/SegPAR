import torch
import numpy as np
import torch.nn.functional as F
from mmseg.apis import inference_model
from adv_setting import model_predict
import math
from MedSAM_Inference_multi_boxes import infer_class_map_from_image_and_boxes


class SegEnv:
    def __init__(self, classification_model, config, image_sizes=None):
        super().__init__()
        self.model = classification_model
        self.pixel = config["attack_pixel"]
        self.action_mask_init = True
        self.config = config
        self.image_sizes = image_sizes
        self.ref_probs = []
        self.ref_classes = []
        self.s = None
        self.ori_init = True
        self.init = True
        self.discrepancy_masks = []
        self.device = config["device"]
        self.all_same_shape = True
        self.action_rgb = []
        self.valid_masks = []
        self.valid_action_masks = []


    def reduce_mask_auto(self, mask_tensor):
        """
        PyTorch       .
        """

        # 1. '1'    (torch.nonzero )
        one_coords = torch.nonzero(mask_tensor, as_tuple=False).float() #    float 
        num_ones = len(one_coords)

        if num_ones == 0:
            print(":  '1'  .")
            return None

        # 2.   m  
        m = int(math.sqrt(num_ones))
        if m == 0:
            print(": 1x1     .")
            return None
            
        num_to_select = m * m
        original_height, original_width = mask_tensor.shape
        

        center_point = torch.tensor([
            (original_height - 1) / 2.0, 
            (original_width - 1) / 2.0
        ], device=mask_tensor.device)
        
        # 4.   '1'   (torch.linalg.norm )
        distances = torch.linalg.norm(one_coords - center_point, axis=1)
        
        # 5.     m*m  (torch.argsort )
        sorted_indices = torch.argsort(distances)
        selected_coords = one_coords[sorted_indices][:num_to_select].long() #   long() 
        

        encoded_values = selected_coords[:, 0] * original_width + selected_coords[:, 1]
        

        encoded_values, _ = torch.sort(encoded_values)
        
        # 8. m x m    
        reduced_mask = encoded_values.reshape((m, m))        
        return reduced_mask


    def decode_mask(self, encoded_values, original_width):
        """
              

        Args:
            encoded_values (np.ndarray):    1D  2D .
            original_width (int):   ( ).

        Returns:
            np.ndarray:  (, )  2D .
        """


        rows = encoded_values // original_width
        cols = encoded_values % original_width
    

        decoded_coords = np.vstack((rows, cols)).T
        
        return decoded_coords


    def step(self, original_images, actions, start_idx, end_idx, attack_pixels, gt_list, benign_pred_list = None):

        if self.config["action_space"] == "standard":
            return self.seg_step(original_images, actions, start_idx, end_idx, attack_pixels, gt_list, benign_pred_list)
        elif self.config["action_space"] == "reg":
            return self.reg_step(original_images, actions, start_idx, end_idx, attack_pixels, gt_list, benign_pred_list)
        else:
            raise ValueError(f"Invalid action space: {self.config['action_space']}")


    def reg_step(self, changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list, benign_pred_list = None, bboxes_list = None, class_ids_list = None):
        return self._reg_step(changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list, benign_pred_list, bboxes_list , class_ids_list )

    def seg_step(self, original_images, actions,
                 start_idx, end_idx, attack_pixels, gt_list, benign_pred_list = None):
        if self.all_same_shape:
            return self._seg_step_same_shape(
                original_images, actions,
                start_idx, end_idx,
                attack_pixels, gt_list, benign_pred_list
            )
        else:
            return self._seg_step_diff_shape(
                original_images, actions,
                start_idx, end_idx,
                attack_pixels, gt_list, benign_pred_list
            )
    
    #### original ####

    def make_transformed_images(self, original_images, actions_list, attack_pixels, action_rgb = None):
        if self.all_same_shape:
            
            # GPU  
            #     (B, H, W, C)
            batched_images = torch.from_numpy(np.stack(original_images)).to(self.device)
            changed_images = batched_images.to(torch.uint8).clone() if self.config["task"] in ["object_detection", "segmentation"] else batched_images.clone()
            
            # actions  
            if isinstance(actions_list, list):
                batched_actions = torch.stack(actions_list).to(self.device)
            else:
                batched_actions = actions_list.to(self.device)
            batched_actions = torch.sigmoid(batched_actions)
            

            x_bound, y_bound = original_images[0].shape[0], original_images[0].shape[1]
            x = (x_bound * batched_actions[..., 0] - 1).long()  # (B, N)
            y = (y_bound * batched_actions[..., 1] - 1).long()  # (B, N)
            
            # RGB   ( )
            if self.config["majority"] is not None:
                # action_rgb   
                action_rgb_stack = torch.stack(action_rgb).to(self.device)  # [B, 2, 3]
                # batched_actions (0  1)  minority  majority RGB 
                indices = (batched_actions[..., 2:5] > 0.5).any(dim=-1, keepdim=True)  # [B, N, 1]
                selected_rgb = action_rgb_stack[torch.arange(len(action_rgb_stack)).unsqueeze(1), indices.long().squeeze(-1)]  # [B, N, 3]
                rgb = selected_rgb.to(torch.uint8)
            else:
                rgb = ((batched_actions[..., 2:5] > 0.5) * (255 if self.config["task"] in ["object_detection", "segmentation"] else 1.0)).to(torch.uint8)
            

            changed_images[torch.arange(len(original_images)).unsqueeze(1), x, y] = rgb
            
            # numpy   
            return list(changed_images.cpu().numpy())
        
        else:


            changed_images = []
            
            for i in range(len(original_images)):

                changed_image = np.copy(original_images[i])
                
                #   action  
                current_action = torch.sigmoid(actions_list[i])
                

                if len(current_action.shape) == 1:
                    current_action = current_action.unsqueeze(0)
                

                x_bound = original_images[i].shape[0]
                y_bound = original_images[i].shape[1]
                

                x = (x_bound * current_action[:, 0] - 1).long()
                y = (y_bound * current_action[:, 1] - 1).long()
                
                # RGB  
                if self.config["majority"] is not None:
                    # action_rgb     
                    current_rgb = action_rgb[i].to(self.device)
                    indices = (current_action[:, 2:5] > 0.5).any(dim=-1, keepdim=True)  # [N, 1]
                    rgb = current_rgb[indices.long().squeeze(-1)]  # [N, 3]
                    rgb = rgb.to(torch.uint8)
                else:
                    rgb = ((current_action[:, 2:5] > 0.5).float() * (255 if self.config["task"] in ["object_detection", "segmentation"] else 1.0)).to(torch.uint8)
                

                n_pixels = attack_pixels[i]
                
                if n_pixels == 1:

                    changed_image[x[0], y[0]] = rgb[0].cpu().numpy()
                else:

                    idx = torch.arange(n_pixels)
                    changed_image[x[idx], y[idx]] = rgb[idx].cpu().numpy()
                
                changed_images.append(changed_image)
            
            return changed_images


    def make_transformed_images_valid_mask(self, original_images, actions_list, attack_pixels, action_rgb = None, valid_action_mask = None):


        


        changed_images = []



        for i in range(len(original_images)):

            changed_image = np.copy(original_images[i])
            h,w = changed_image.shape[:2]
            
            #   action  
            current_action = torch.sigmoid(actions_list[i])
            

            if len(current_action.shape) == 1:
                current_action = current_action.unsqueeze(0)
            

            x_bound = valid_action_mask[i].shape[1]
            y_bound = valid_action_mask[i].shape[0]
            

            x = (x_bound * current_action[:, 0] - 1).long()
            y = (y_bound * current_action[:, 1] - 1).long()
            
            # RGB  
            if self.config["majority"] is not None:
                # action_rgb     
                current_rgb = action_rgb[i].to(self.device)
                indices = (current_action[:, 2:5] > 0.5).any(dim=-1, keepdim=True)  # [N, 1]
                rgb = current_rgb[indices.long().squeeze(-1)]  # [N, 3]
                rgb = rgb.to(torch.uint8)
            else:
                rgb = ((current_action[:, 2:5] > 0.5).float() * (255 if self.config["task"] in ["object_detection", "segmentation"] else 1.0)).to(torch.uint8)
            

            n_pixels = attack_pixels[i]

            # extract valid action position
            encoded_action_list = []
            for j in range(len(x)):
                encoded_action_list.append(valid_action_mask[i][x[j], y[j]].item())
            encoded_action_list = np.array(encoded_action_list)

            # decode valid action position
            valid_action_position = self.decode_mask(encoded_action_list, w)

            

            if n_pixels == 1:

                changed_image[valid_action_position[0]] = rgb[0].cpu().numpy()
            else:

                idx = torch.arange(n_pixels)

                changed_image[valid_action_position[idx,0],valid_action_position[idx,1]] = rgb[idx].cpu().numpy()



            changed_images.append(changed_image)

        
        return changed_images


    def _seg_initial_step(self, original_images, gt_tensors, benign_pred_tensors, bbox_list = None, class_ids_list =None):

        for i, (img, gt) in enumerate(zip(original_images, gt_tensors)):
            if self.config["dataset"] == "ade20k" or self.config["dataset"] == "VOC2012" or self.config["dataset"] == "cityscapes":
                res  = inference_model(self.model, img)
                soft = F.softmax(res.seg_logits.data.squeeze(), 0)
                pred = res.pred_sem_seg.data.squeeze()      




                ref_p = torch.gather(soft, 0, pred.unsqueeze(0)).squeeze(0) # (H,W)
                self.ref_probs.append(ref_p)


            elif self.config["dataset"] == "CT_Abd":
                pred = infer_class_map_from_image_and_boxes(self.model, img, bbox_list[i], class_ids_list[i])
            self.ref_classes.append(pred.clone())
            benign_pred_tensors[i].copy_(pred)


            if self.ori_init and self.config["majority"] is not None:
                if self.config["majority"] == "Majority":
                    rgb = self.compute_region_channel_means_majority(img, pred)
                else:
                    rgb = self.compute_region_channel_means_minority(img, pred)
                self.action_rgb.append(rgb)  # rgb: Tensor(3,)

            gt_t = torch.from_numpy(gt).to(self.device) \
                if not isinstance(gt, torch.Tensor) else gt.to(self.device)

            if self.config["dataset"] == "ade20k":
                if self.config["use_gt"] == True:
                    discrepancy_mask = (pred != (gt_t-1)).int()
                    vmask =(pred == (gt_t-1)).int()
                else:

                    benign_pred = benign_pred_tensors[i]
                    
                    discrepancy_mask = (pred != benign_pred).int()
                    # new
                    vmask = (pred == benign_pred).int() * (gt_t!=0).int()
                    # old
                    # vmask = (gt_t!=0).int()
            elif self.config["dataset"] == "VOC2012":
                if self.config["use_gt"] == True:
                    discrepancy_mask = (pred != gt_t).int()
                    vmask =(pred == gt_t).int()
                else:
                    discrepancy_mask = (pred != benign_pred_tensors[i]).int()
                    # vmask = (pred == benign_pred_list[i]).int()
                    vmask = (pred == benign_pred_tensors[i]).int() *  (gt_t!=255).int()
            elif self.config["dataset"] == "cityscapes":
                if self.config["use_gt"] == True:
                    discrepancy_mask = (pred != gt_t).int()
                    vmask =(pred == gt_t).int()
                else:
                    
                    discrepancy_mask = (pred != benign_pred_tensors[i]).int()
                    # new
                    vmask = (pred == benign_pred_tensors[i]).int() * (gt_t!=255).int()
            elif self.config["dataset"] == "CT_Abd":
                if self.config["use_gt"] == True:
                    discrepancy_mask = (pred != gt_t).int()
                    vmask =(pred == gt_t).int()
                else:
                    
                    discrepancy_mask = (pred != benign_pred_tensors[i]).int()
                    # new
                    vmask = (pred == benign_pred_tensors[i]).int() * (gt_t!=255).int()


            self.valid_masks.append(vmask)
            self.discrepancy_masks.append(discrepancy_mask)


    

    # ────────── (1)     (list → stack) ──────────
    def _seg_step_same_shape(self, original_images, actions,
                             start_idx, end_idx, attack_pixels, gt_tensors, benign_pred_tensors):
        with torch.no_grad():

            if self.init:
                results = inference_model(self.model, original_images)         # list[B]
                seg_logits = torch.stack([r.seg_logits.data for r in results]) # (B,C,H,W)
                pred_sem   = torch.stack([r.pred_sem_seg.data.squeeze() for r in results])
                probs_ref  = torch.gather(
                    F.softmax(seg_logits, 1), 1, pred_sem.unsqueeze(1)
                ).squeeze(1)                                                        # (B,H,W)

                # list  extend
                sl = slice(start_idx, end_idx)
                self.ref_probs  [sl].copy_(probs_ref)
                self.ref_classes[sl].copy_(pred_sem)
                benign_pred_tensors[:len(actions)].copy_(pred_sem)

                # majority / minority
                if self.config["majority"] is not None:
                    fn = (self.compute_region_channel_means_majority
                          if self.config["majority"] == "Majority"
                          else self.compute_region_channel_means_minority)
                    rgb_batch = fn(original_images, pred_sem)   # (B',3)
                    self.action_rgb[sl].copy_(rgb_batch)

                # valid-mask

                if self.config["dataset"] == "cityscapes":
                    if self.config["use_gt"] == True:
                        discrepancy_mask = (pred_sem != gt_tensors).int()
                        vmask =(pred_sem == gt_tensors).int()
                    else:
                        
                        discrepancy_mask = (pred_sem != benign_pred_tensors).int()
                        # new
                        vmask = (pred_sem == benign_pred_tensors).int() * (gt_tensors!=255).int()

                self.discrepancy_masks[sl].copy_(discrepancy_mask)
                self.valid_masks[sl].copy_(vmask)

                if self.config["update_valid_action"] == True:
                    if self.action_mask_init == True:
                        for idx in range(len(original_images)):
                            vmask = self.valid_masks[start_idx+idx]
                            self.valid_action_masks.append(self.reduce_mask_auto(vmask))
                        


            rgb_slice = None
            if self.config["majority"] is not None:
                rgb_slice = self.action_rgb[start_idx:end_idx]
                
            if self.config["update_valid_action"] == True:
                changed_images= self.make_transformed_images_valid_mask(
                    original_images, actions, attack_pixels, rgb_slice, self.valid_action_masks[start_idx:end_idx]
                )
                attacked_pixels = attack_pixels
            else:

                changed_images = self.make_transformed_images(
                    original_images, actions, attack_pixels, rgb_slice
                )
                attacked_pixels = attack_pixels


            # (2)  – reward   (ref_*  )


            ref_p  = self.ref_probs  [start_idx:end_idx].to(self.device)
            
            vmask  = self.valid_masks [start_idx:end_idx].to(self.device)

            res_ch  = inference_model(self.model, changed_images)
            logits_ch = torch.stack([r.seg_logits.data for r in res_ch])
            pred_ch   = torch.stack([r.pred_sem_seg.data.squeeze() for r in res_ch])
            probs_ch  = torch.gather(
                F.softmax(logits_ch, 1), 1, benign_pred_tensors.type(torch.int64).unsqueeze(1)
            ).squeeze(1)

            # w =1 , confidence
            # w =0 , decision
            # w =0.5 , original_combined

            if self.config["reward_type"] == "standard" or self.config["reward_type"] == "reduction":
                ref_c  = self.ref_classes[start_idx:end_idx].to(self.device)
                rewards = (
                    self.config["w"]*(((ref_p - probs_ch)*vmask).sum(dim=(1,2)) / vmask.sum(dim=(1,2))) +
                    (1-self.config["w"])*(((pred_ch != ref_c)*vmask).sum(dim=(1,2)) /
                    (torch.tensor(attacked_pixels, device=self.device) *
                    self.config["baseline"] ** 2))
                )

            elif self.config["reward_type"] == "discrepancy":
                discrepancy_mask = self.discrepancy_masks[start_idx:end_idx].to(self.device)

                if self.config["use_gt"] == True:
                    discrepancy_mask_new = (pred_ch != (gt_tensors)).int()
                else:
                    discrepancy_mask_new = (pred_ch != benign_pred_tensors.to(self.device)).int()
                
                temp_mat = (discrepancy_mask_new - discrepancy_mask)*vmask     
                ats_pix = (temp_mat>0).sum(dim=(1,2))
                change_pix = (temp_mat<0).sum(dim=(1,2))
                
                rewards = (
                    self.config["w"]*(((ref_p - probs_ch)*vmask).sum(dim=(1,2)) / vmask.sum(dim=(1,2))) +
                    (1-self.config["w"])*(ats_pix-change_pix) /
                    (torch.tensor(attacked_pixels, device=self.device) *
                    self.config["baseline"] ** 2)
                )
            



        if self.config["reward_type"] == "standard" or self.config["reward_type"] == "reduction":

            return rewards, changed_images, probs_ch, pred_ch
        elif self.config["reward_type"] == "discrepancy":
            return rewards, changed_images, probs_ch, discrepancy_mask_new

    # ────────── (2)     (list ) ──────────
    def _seg_step_diff_shape(self, original_images, actions,
                             start_idx, end_idx, attack_pixels, gt_list, benign_pred_list = None):
        with torch.no_grad():

            if self.init:
                for i, (img, gt) in enumerate(zip(original_images, gt_list)):
                    res  = inference_model(self.model, img)
                    soft = F.softmax(res.seg_logits.data.squeeze(), 0)
                    pred = res.pred_sem_seg.data.squeeze()                                # (H,W)



                    ref_p = torch.gather(soft, 0, pred.unsqueeze(0)).squeeze(0) # (H,W)
                    self.ref_probs.append(ref_p)
                    self.ref_classes.append(pred.clone())
                    benign_pred_list[i].copy_(pred)


                    if self.ori_init and self.config["majority"] is not None:
                        if self.config["majority"] == "Majority":
                            rgb = self.compute_region_channel_means_majority(img, pred)
                        else:
                            rgb = self.compute_region_channel_means_minority(img, pred)
                        self.action_rgb.append(rgb)  # rgb: Tensor(3,)

                    gt_t = torch.from_numpy(gt).to(self.device) \
                           if not isinstance(gt, torch.Tensor) else gt.to(self.device)
                    if self.config["dataset"] == "ade20k":
                        if self.config["use_gt"] == True:
                            discrepancy_mask = (pred != (gt_t-1)).int()
                            vmask =(pred == (gt_t-1)).int()
                        else:

                            benign_pred = benign_pred_list[i]
                            
                            discrepancy_mask = (pred != benign_pred).int()
                            # new
                            vmask = (pred == benign_pred).int() * (gt_t!=0).int()
                            # old
                            # vmask = (gt_t!=0).int()
                    elif self.config["dataset"] == "VOC2012":
                        if self.config["use_gt"] == True:
                            discrepancy_mask = (pred != gt_t).int()
                            vmask =(pred == gt_t).int()
                        else:
                            discrepancy_mask = (pred != benign_pred_list[i]).int()
                            # vmask = (pred == benign_pred_list[i]).int()
                            vmask = (pred == benign_pred_list[i]).int() *  (gt_t!=255).int()
                    else:
                        # custom
                        discrepancy_mask = (pred == gt_t).int()


                    self.valid_masks.append(vmask)
                    self.discrepancy_masks.append(discrepancy_mask)

                if self.config["update_valid_action"] == True:
                    if self.action_mask_init == True:
                        for idx in range(len(original_images)):
                            vmask = self.valid_masks[start_idx+idx]
                            self.valid_action_masks.append(self.reduce_mask_auto(vmask))


            rgb_slice = None
            if self.config["majority"] is not None:
                rgb_slice = self.action_rgb[start_idx:end_idx]


            if self.config["update_valid_action"] == True:
                changed_images = self.make_transformed_images_valid_mask(
                    original_images, actions, attack_pixels, rgb_slice, self.valid_action_masks[start_idx:end_idx]
                )
                attacked_pixels = attack_pixels

            else:
                changed_images = self.make_transformed_images(
                    original_images, actions, attack_pixels, rgb_slice
                )
                attacked_pixels = attack_pixels



            # ③ reward  ( )
            rewards = []
            discrepancy_mask_list = []
            probs_ch_list = []
            pred_ch_list = []

            if self.config["reward_type"] == "standard" or self.config["reward_type"] == "reduction":
                for i, (img, atk_pix) in enumerate(zip(changed_images, attacked_pixels)):
                    ref_p = self.ref_probs  [start_idx + i]
                    ref_c = self.ref_classes[start_idx + i].long()  # int64 
                    vmask = self.valid_masks [start_idx + i]

                    res   = inference_model(self.model, img)
                    soft  = F.softmax(res.seg_logits.data.squeeze(), 0)
                    pred  = res.pred_sem_seg.data.squeeze()                                   # (H,W)
                    probs = torch.gather(soft, 0, benign_pred_list[i].type(torch.int64).unsqueeze(0)).squeeze(0)      # (H,W)


                    
                    gt_t = torch.from_numpy(gt_list[i]).to(self.device) 



                    r_mat   = (ref_p - probs) * vmask
                    change_cls = (pred != ref_c)*vmask

                    # w =1 , confidence
                    # w =0 , decision
                    # w =0.5 , original_combined


                    reward = (
                            self.config["w"]*(r_mat.sum() / vmask.sum()) +
                            (1-self.config["w"])*(change_cls.sum() /
                            (torch.tensor(atk_pix, device=self.device) *
                            self.config["baseline"] ** 2))
                        )

                    probs_ch_list.append(probs)
                    pred_ch_list.append(pred)
                    rewards.append(reward)
                    

                rewards = torch.tensor(rewards, device=self.device)


                return rewards, changed_images, probs_ch_list, pred_ch_list

            elif self.config["reward_type"] == "discrepancy":


                for i, (img, atk_pix) in enumerate(zip(changed_images, attacked_pixels)):
                    ref_p = self.ref_probs  [start_idx + i]
                    ref_c = self.ref_classes[start_idx + i].long()  # int64 
                    vmask = self.valid_masks [start_idx + i]
                    discrepancy_mask = self.discrepancy_masks[start_idx + i]

                    res   = inference_model(self.model, img)
                    soft  = F.softmax(res.seg_logits.data.squeeze(), 0)
                    pred  = res.pred_sem_seg.data.squeeze()                                   # (H,W)
                    probs = torch.gather(soft, 0, benign_pred_list[i].type(torch.int64).unsqueeze(0)).squeeze(0)      # (H,W)


                    
                    gt_t = torch.from_numpy(gt_list[i]).to(self.device) 

                    r_mat   = (ref_p - probs) * vmask

                    if self.config["use_gt"] == True:
                        discrepancy_mask_new = (pred != (gt_t-1)).int()
                    else:
                        benign_pred = benign_pred_list[i]
                        discrepancy_mask_new = (pred != benign_pred).int()

                    temp_mat = (discrepancy_mask_new - discrepancy_mask)*vmask     
                    ats_pix = (temp_mat>0).sum()
                    change_pix = (temp_mat<0).sum()


                    # w =1 , confidence
                    # w =0 , decision
                    # w =0.5 , original_combined



                    reward = (
                        self.config["w"]*(r_mat.sum() / vmask.sum()) +
                        (1-self.config["w"])*(ats_pix-change_pix) /
                        (torch.tensor(atk_pix, device=self.device) *
                        self.config["baseline"] ** 2)
                    )


                    # reward = (
                    #     (r_mat.sum() / vmask.sum()) +
                    #     (ats_pix/(self.factor*change_pix+1)) /
                    #     (torch.tensor(atk_pix, device=self.device) *
                    #     self.config["baseline"] ** 2)

                    
            
                    discrepancy_mask_list.append(discrepancy_mask_new)
                    probs_ch_list.append(probs)
                    pred_ch_list.append(pred)
                    rewards.append(reward)    
                rewards = torch.tensor(rewards, device=self.device)

                return rewards, changed_images, probs_ch_list, discrepancy_mask_list


    def _reg_step(self, changed_images, start_idx, end_idx, batch_attack_mask_set, meta_list, benign_pred_list = None, bboxes_list = None, class_ids_list = None):
        with torch.no_grad():
        # ③ reward  ( )
            rewards = []
            memory_reward = []
            discrepancy_mask_list = []
            pred_ch_list = []

            
                

            if self.config["reward_type"] == "standard" or self.config["reward_type"] == "reduction":
                for i in range(len(changed_images)):
                    img = changed_images[i]
                    meta = meta_list[i]
                    attack_mask = batch_attack_mask_set[i]

                    ref_c = self.ref_classes[start_idx + i].long()  # int64 
                    vmask = self.valid_masks [start_idx + i]

                    if self.config["dataset"] == "CT_Abd":
                        bbox = bboxes_list[i]
                        class_ids = class_ids_list[i]
                        pred = infer_class_map_from_image_and_boxes(self.model, img, bbox, class_ids)
                    else:

                        res   = inference_model(self.model, img)
                        pred  = res.pred_sem_seg.data.squeeze()  
                    change_cls = (pred != ref_c)*vmask

                    temp_reward = 0

                    for j in range(len(meta)):
                        benign_mask = attack_mask[j].to(self.device)
                        atk_pix = meta[j]['attack_pixels']
                        regional_cls = change_cls*benign_mask

                        # w =1 , confidence
                        # w =0 , decision
                        # w =0.5 , original_combined


                        if atk_pix == 0:
                            reward = torch.tensor(0.0, device=self.device)
                            
                        else:
                            reward = (
                                    (1-self.config["w"])*(regional_cls.sum() /
                                    (torch.tensor(atk_pix, device=self.device) *
                                    self.config["baseline"] ** 2))
                                    )
                        rewards.append(reward)
                        temp_reward += reward

                    pred_ch_list.append(pred)
                    memory_reward.append(temp_reward/len(meta))
                        
                        

                rewards = torch.tensor(rewards, device=self.device)
                memory_reward = torch.tensor(memory_reward, device=self.device)


                return rewards, changed_images, pred_ch_list, memory_reward

            elif self.config["reward_type"] == "discrepancy":

                for i in range(len(changed_images)):
                    img = changed_images[i]
                    ref_c = self.ref_classes[start_idx + i].long()  # int64 
                    vmask = self.valid_masks [start_idx + i]
                    discrepancy_mask = self.discrepancy_masks[start_idx + i]
                    meta = meta_list[i]
                    attack_mask = batch_attack_mask_set[i]
                    temp_reward = 0

                    if self.config["dataset"] == "CT_Abd":
                        bbox = bboxes_list[i]
                        class_ids = class_ids_list[i]
                        pred = infer_class_map_from_image_and_boxes(self.model, img, bbox, class_ids)
                    else:

                        res   = inference_model(self.model, img)
                        pred  = res.pred_sem_seg.data.squeeze()  

                    if self.config["use_gt"] == True:
                            discrepancy_mask_new = (pred != (gt_t-1)).int()
                    else:
                        benign_pred = benign_pred_list[i]
                        discrepancy_mask_new = (pred != benign_pred).int()


                    for j in range(len(meta)):
                        benign_mask = attack_mask[j].to(self.device)
                        atk_pix = meta[j]['attack_pixels']

                        if atk_pix == 0:
                            reward = torch.tensor(0.0, device=self.device)
                        else:

                            temp_mat = (discrepancy_mask_new - discrepancy_mask)*vmask* benign_mask
                            ats_pix = (temp_mat>0).sum()
                            change_pix = (temp_mat<0).sum()


                            # w =1 , confidence
                            # w =0 , decision
                            # w =0.5 , original_combined



                            reward = (
                                (1-self.config["w"])*(ats_pix-change_pix) /
                                (torch.tensor(atk_pix, device=self.device) *
                                self.config["baseline"] ** 2)
                            )

                
                        
                        rewards.append(reward)    
                        temp_reward += reward
                    memory_reward.append(temp_reward/len(meta))
                    pred_ch_list.append(pred)
                    discrepancy_mask_list.append(discrepancy_mask_new)
                rewards = torch.tensor(rewards, device=self.device)
                memory_reward = torch.tensor(memory_reward, device=self.device)

                return rewards, changed_images, pred_ch_list, discrepancy_mask_list, memory_reward
                
    def _adv_reg_initial_step(self, batch_images, batch_gts, benign_pred_tensors=None):
        for i, (img, gt) in enumerate(zip(batch_images, batch_gts)):
            output, pred = model_predict(self.model, img, self.config)
            pred = pred.to(self.device)
            ref_p = torch.gather(output, 0, pred.type(torch.int64).unsqueeze(0)).squeeze(0)

            self.ref_probs.append(ref_p)
            self.ref_classes.append(pred.clone())
            if benign_pred_tensors is not None:
                benign_pred_tensors[i].copy_(pred)

            gt_t = torch.from_numpy(gt).to(self.device) if not isinstance(gt, torch.Tensor) else gt.to(self.device)

            if self.config["dataset"] == "ade20k":
                if self.config["use_gt"] is True:
                    discrepancy_mask = (pred != (gt_t - 1)).int()
                    vmask = (pred == (gt_t - 1)).int()
                else:
                    discrepancy_mask = (pred != benign_pred_tensors[i]).int()
                    vmask = (pred == benign_pred_tensors[i]).int() * (gt_t != 0).int()
            elif self.config["dataset"] == "VOC2012":
                if self.config["use_gt"] is True:
                    discrepancy_mask = (pred != gt_t).int()
                    vmask = (pred == gt_t).int()
                else:
                    discrepancy_mask = (pred != benign_pred_tensors[i]).int()
                    vmask = (pred == benign_pred_tensors[i]).int() * (gt_t != 255).int()
            else:
                if self.config["use_gt"] is True:
                    discrepancy_mask = (pred != gt_t).int()
                    vmask = (pred == gt_t).int()
                else:
                    discrepancy_mask = (pred != benign_pred_tensors[i]).int()
                    vmask = (pred == benign_pred_tensors[i]).int() * (gt_t != 255).int()

            self.valid_masks.append(vmask)
            self.discrepancy_masks.append(discrepancy_mask)

    def _adv_reg_step(self, changed_images, start_idx, batch_attack_mask_set, meta_list):
        rewards = []
        memory_reward = []
        pred_ch_list = []
        discrepancy_mask_list = []

        with torch.no_grad():
            for i in range(len(changed_images)):
                _, pred = model_predict(self.model, changed_images[i], self.config)
                pred = pred.to(self.device)

                ref_c = self.ref_classes[start_idx + i].long().to(self.device)
                vmask = self.valid_masks[start_idx + i].to(self.device)

                temp_reward = torch.tensor(0.0, device=self.device)
                attack_mask = batch_attack_mask_set[i]
                obj_meta = meta_list[i]

                if self.config["reward_type"] == "standard" or self.config["reward_type"] == "reduction":
                    change_cls = (pred != ref_c) * vmask
                    for j in range(len(obj_meta)):
                        benign_mask = attack_mask[j].to(self.device)
                        atk_pix = obj_meta[j]["attack_pixels"]
                        if atk_pix == 0:
                            reward = torch.tensor(0.0, device=self.device)
                        else:
                            reward = (
                                (1 - self.config["w"]) * ((change_cls * benign_mask).sum() /
                                (torch.tensor(atk_pix, device=self.device) * self.config["baseline"] ** 2))
                            )
                        rewards.append(reward)
                        temp_reward += reward
                    pred_ch_list.append(pred)
                    memory_reward.append(temp_reward / max(1, len(obj_meta)))
                else:
                    discrepancy_mask = self.discrepancy_masks[start_idx + i].to(self.device)
                    discrepancy_mask_new = (pred != ref_c).int()
                    temp_mat = (discrepancy_mask_new - discrepancy_mask) * vmask

                    for j in range(len(obj_meta)):
                        benign_mask = attack_mask[j].to(self.device)
                        atk_pix = obj_meta[j]["attack_pixels"]
                        if atk_pix == 0:
                            reward = torch.tensor(0.0, device=self.device)
                        else:
                            reg_ats_pix = ((temp_mat * benign_mask) > 0).sum()
                            reg_change_pix = ((temp_mat * benign_mask) < 0).sum()
                            reward = (
                                (1 - self.config["w"]) * (reg_ats_pix - reg_change_pix) /
                                (torch.tensor(atk_pix, device=self.device) * self.config["baseline"] ** 2)
                            )
                        rewards.append(reward)
                        temp_reward += reward
                    pred_ch_list.append(pred)
                    discrepancy_mask_list.append(discrepancy_mask_new)
                    memory_reward.append(temp_reward / max(1, len(obj_meta)))

        rewards = torch.stack(rewards) if rewards else torch.tensor([], device=self.device)
        memory_reward = torch.stack(memory_reward) if memory_reward else torch.tensor([], device=self.device)

        if self.config["reward_type"] == "standard" or self.config["reward_type"] == "reduction":
            return rewards, changed_images, pred_ch_list, memory_reward
        return rewards, changed_images, pred_ch_list, discrepancy_mask_list, memory_reward


    def adv_setting_step(self, original_images, actions, start_idx, end_idx, attack_pixels, gt_list, benign_pred_list = None):

        return self.adv_setting_seg_step(original_images, actions, start_idx, end_idx, attack_pixels, gt_list, benign_pred_list)

    def adv_setting_seg_step(self, original_images, actions,
                 start_idx, end_idx, attack_pixels, gt_list, benign_pred_list = None):
        if self.all_same_shape:
            return self._adv_setting_seg_step_same_shape(
                original_images, actions,
                start_idx, end_idx,
                attack_pixels, gt_list, benign_pred_list
            )
        else:
            return self._adv_setting_seg_step_diff_shape(
                original_images, actions,
                start_idx, end_idx,
                attack_pixels, gt_list, benign_pred_list
            )

    # ────────── (1)     (list → stack) ──────────
    def _adv_setting_seg_step_same_shape(self, original_images, actions,
                             start_idx, end_idx, attack_pixels, gt_tensors, benign_pred_tensors):
        with torch.no_grad():

            if self.init:
                for i, (img, gt) in enumerate(zip(original_images, gt_tensors)):
                    output, prediction = model_predict(self.model, img, self.config)
                    seg_conf = output # (1,C,H,W)
                    pred_sem   = prediction # (H,W)
                    probs_ref  = torch.gather(
                        seg_conf, 0, pred_sem.type(torch.int64).unsqueeze(0)
                    ).squeeze()                                                        # (B,H,W)

                    if i == 0:
                        total_probs_ref = probs_ref.unsqueeze(0)
                        total_pred_sem = pred_sem.unsqueeze(0)
                    else:

                        total_probs_ref = torch.cat((total_probs_ref, probs_ref.unsqueeze(0)), dim=0)
                        total_pred_sem = torch.cat((total_pred_sem, pred_sem.unsqueeze(0)), dim=0)


                # list  extend
                sl = slice(start_idx, end_idx)
                self.ref_probs  [sl].copy_(total_probs_ref)
                self.ref_classes[sl].copy_(total_pred_sem)
                benign_pred_tensors[:len(actions)].copy_(total_pred_sem)

                # majority / minority
                if self.config["majority"] is not None:
                    fn = (self.compute_region_channel_means_majority
                          if self.config["majority"] == "Majority"
                          else self.compute_region_channel_means_minority)
                    rgb_batch = fn(original_images, pred_sem)   # (B',3)
                    self.action_rgb[sl].copy_(rgb_batch)

                # valid-mask

                if self.config["dataset"] == "cityscapes":
                    if self.config["use_gt"] == True:
                        discrepancy_mask = (pred_sem != gt_tensors).int()
                        vmask =(pred_sem == gt_tensors).int()
                    else:
                        
                        discrepancy_mask = (pred_sem != benign_pred_tensors).int()
                        # new
                        vmask = (pred_sem == benign_pred_tensors).int() * (gt_tensors!=255).int()

                self.discrepancy_masks[sl].copy_(discrepancy_mask)
                self.valid_masks[sl].copy_(vmask)


            rgb_slice = None
            if self.config["majority"] is not None:
                rgb_slice = self.action_rgb[start_idx:end_idx]

            changed_images = self.make_transformed_images(
                original_images, actions, attack_pixels, rgb_slice
            )


            # (2)  – reward   (ref_*  )


            ref_p  = self.ref_probs  [start_idx:end_idx].to(self.device)
            vmask  = self.valid_masks [start_idx:end_idx].to(self.device)

            for i, (img, atk_pix) in enumerate(zip(changed_images, attack_pixels)):
                output, prediction = model_predict(self.model, img, self.config)

                probs_ch  = torch.gather(
                    output, 0, benign_pred_tensors[i].type(torch.int64).unsqueeze(0)
                ).squeeze()

                if i == 0:
                    total_probs_ch = probs_ch.unsqueeze(0)
                    total_pred_ch = prediction.unsqueeze(0)
                else:
                    total_probs_ch = torch.cat((total_probs_ch, probs_ch.unsqueeze(0)), dim=0)
                    total_pred_ch = torch.cat((total_pred_ch, prediction.unsqueeze(0)), dim=0)
            


            

            if self.config["reward_type"] == "standard":
                ref_c  = self.ref_classes[start_idx:end_idx].to(self.device)
                rewards = (
                    (((ref_p - total_probs_ch)*vmask).sum(dim=(1,2)) / vmask.sum(dim=(1,2))) +
                    (((total_pred_ch != ref_c)*vmask).sum(dim=(1,2)) /
                    (torch.tensor(attack_pixels, device=self.device) *
                    self.config["baseline"] ** 2))
                )

            elif self.config["reward_type"] == "discrepancy":
                discrepancy_mask = self.discrepancy_masks[start_idx:end_idx].to(self.device)

                if self.config["use_gt"] == True:
                    discrepancy_mask_new = (total_pred_ch != (gt_tensors)).int()
                else:
                    discrepancy_mask_new = (total_pred_ch != benign_pred_tensors.to(self.device)).int()
                
                temp_mat = (discrepancy_mask_new - discrepancy_mask)*vmask     
                ats_pix = (temp_mat>0).sum(dim=(1,2))
                change_pix = (temp_mat<0).sum(dim=(1,2))
                
                rewards = (
                    (((ref_p - total_probs_ch)*vmask).sum(dim=(1,2)) / vmask.sum(dim=(1,2))) +
                    (ats_pix-change_pix) /
                    (torch.tensor(attack_pixels, device=self.device) *
                    self.config["baseline"] ** 2)
                )
 
        if self.config["reward_type"] == "standard":
            return rewards, changed_images, total_probs_ch, total_pred_ch
        else:
            return rewards, changed_images, total_probs_ch, discrepancy_mask_new

    # ────────── (2)     (list ) ──────────
    def _adv_setting_seg_step_diff_shape(self, original_images, actions,
                             start_idx, end_idx, attack_pixels, gt_list, benign_pred_list = None):
        with torch.no_grad():

            if self.init:
                for i, (img, gt) in enumerate(zip(original_images, gt_list)):

                    output, prediction = model_predict(self.model, img, self.config)



                    ref_p = torch.gather(output, 0, prediction.unsqueeze(0).type(torch.int64)).squeeze(0) # (H,W)
                    self.ref_probs.append(ref_p)
                    self.ref_classes.append(prediction.clone())
                    benign_pred_list[i].copy_(prediction)


                    if self.ori_init and self.config["majority"] is not None:
                        if self.config["majority"] == "Majority":
                            rgb = self.compute_region_channel_means_majority(img, prediction)
                        else:
                            rgb = self.compute_region_channel_means_minority(img, prediction)
                        self.action_rgb.append(rgb)  # rgb: Tensor(3,)

                    gt_t = torch.from_numpy(gt).to(self.device) \
                           if not isinstance(gt, torch.Tensor) else gt.to(self.device)
                    if self.config["dataset"] == "ade20k":
                        if self.config["use_gt"] == True:
                            discrepancy_mask = (prediction != (gt_t-1)).int()
                            vmask =(prediction == (gt_t-1)).int()
                        else:

                            benign_pred = benign_pred_list[i]
                            
                            discrepancy_mask = (prediction != benign_pred).int()
                            # new
                            vmask = (prediction == benign_pred).int() * (gt_t!=0).int()
                            # old
                            # vmask = (gt_t!=0).int()
                    elif self.config["dataset"] == "VOC2012":
                        if self.config["use_gt"] == True:
                            discrepancy_mask = (prediction != gt_t).int()
                            vmask =(prediction == gt_t).int()
                        else:
  
                            discrepancy_mask = (prediction   != benign_pred_list[i]).int()
                            vmask = (prediction == benign_pred_list[i]).int() * ((gt_t!=0) | (gt_t!=255)).int()
                    else:
                        # custom
                        discrepancy_mask = (prediction == gt_t).int()


                    self.valid_masks.append(vmask)
                    self.discrepancy_masks.append(discrepancy_mask)


            rgb_slice = None
            if self.config["majority"] is not None:
                rgb_slice = self.action_rgb[start_idx:end_idx]

            changed_images = self.make_transformed_images(
                original_images, actions, attack_pixels, rgb_slice
            )

            # ③ reward  ( )
            rewards = []
            discrepancy_mask_list = []
            probs_ch_list = []
            pred_ch_list = []

            if self.config["reward_type"] == "standard":
                for i, (img, atk_pix) in enumerate(zip(changed_images, attack_pixels)):
                    ref_p = self.ref_probs  [start_idx + i]
                    ref_c = self.ref_classes[start_idx + i].long()  # int64 
                    vmask = self.valid_masks [start_idx + i]

                    output, prediction = model_predict(self.model, img, self.config)

                    soft  = output.squeeze()                             #(C,H,W)                                   
                    probs = torch.gather(soft, 0, benign_pred_list[i].type(torch.int64).unsqueeze(0)).squeeze(0)      # (H,W)



                    r_mat   = (ref_p - probs) * vmask
                    change_cls = (prediction != ref_c)*vmask

                    reward = (
                            (r_mat.sum() / vmask.sum()) +
                            (change_cls.sum() /
                            (torch.tensor(atk_pix, device=self.device) *
                            self.config["baseline"] ** 2))
                        )
                    probs_ch_list.append(probs)
                    pred_ch_list.append(prediction)
                    rewards.append(reward)
                    

                rewards = torch.tensor(rewards, device=self.device)

                return rewards, changed_images, probs_ch_list, pred_ch_list

            elif self.config["reward_type"] == "discrepancy":


                for i, (img, atk_pix) in enumerate(zip(changed_images, attack_pixels)):
                    ref_p = self.ref_probs  [start_idx + i]
                    ref_c = self.ref_classes[start_idx + i].long()  # int64 
                    vmask = self.valid_masks [start_idx + i]
                    discrepancy_mask = self.discrepancy_masks[start_idx + i]

                    output, prediction = model_predict(self.model, img, self.config)
                    
                    soft  = output.squeeze()                             #(C,H,W)
                    
                    probs = torch.gather(soft, 0, benign_pred_list[i].type(torch.int64).unsqueeze(0)).squeeze(0)      # (H,W)


                    
                    

                    r_mat   = (ref_p - probs) * vmask

                    if self.config["use_gt"] == True:
                        gt_t = torch.from_numpy(gt_list[i]).to(self.device) 
                        discrepancy_mask_new = (prediction != (gt_t-1)).int()
                    else:
                        benign_pred = benign_pred_list[i]
                        discrepancy_mask_new = (prediction != benign_pred).int()

                    temp_mat = (discrepancy_mask_new - discrepancy_mask)*vmask     
                    ats_pix = (temp_mat>0).sum()
                    change_pix = (temp_mat<0).sum()





                    
                    reward = (
                        (r_mat.sum() / vmask.sum()) +
                        (ats_pix-change_pix) /
                        (torch.tensor(atk_pix, device=self.device) *
                        self.config["baseline"] ** 2)
                    )
                    
            
                    discrepancy_mask_list.append(discrepancy_mask_new)
                    probs_ch_list.append(probs)
                    pred_ch_list.append(prediction)
                    rewards.append(reward)    
                rewards = torch.tensor(rewards, device=self.device)

                return rewards, changed_images, probs_ch_list, discrepancy_mask_list
                

    def compute_region_channel_means_majority(self, benign_images, segmentation_masks):
        """
        benign_images: List[np.ndarray],   (H, W, 3)   
        segmentation_masks: List[torch.Tensor],   (H, W)  segmentation mask 
        
        segmentation mask     (1st majority)     (2nd majority) ,
        benign image      (R, G, B)   tensor .
        
        Returns:
             shape (2, 3) 
                row: [R_mean, G_mean, B_mean] (1st majority class )
                row: [R_mean, G_mean, B_mean] (2nd majority class )
        """

        
        for benign_img, segmentation_mask in zip(benign_images, segmentation_masks):
            #    GPU 
            benign_img = torch.from_numpy(benign_img.copy()).to(self.device)  # (H, W, 3)
            segmentation_mask = segmentation_mask.to(self.device)  # (H, W)
            

            unique_vals, counts = torch.unique(segmentation_mask, return_counts=True)
            
            # 1st 2nd majority  
            sorted_indices = torch.argsort(counts, descending=True)
            first_majority_class = unique_vals[sorted_indices[0]]
            second_majority_class = unique_vals[sorted_indices[1]] if len(sorted_indices) > 1 else first_majority_class
            

            first_majority_mask = (segmentation_mask == first_majority_class)  # (H, W)
            second_majority_mask = (segmentation_mask == second_majority_class)  # (H, W)
            

            if first_majority_mask.sum() > 0:
                means_first = benign_img[first_majority_mask].float().mean(dim=0)
            else:
                means_first = torch.zeros(3, device=self.device)
                
            if second_majority_mask.sum() > 0:
                means_second = benign_img[second_majority_mask].float().mean(dim=0)
            else:
                means_second = torch.zeros(3, device=self.device)
            

            result_tensor = torch.stack([means_first, means_second])
        
        return result_tensor

    def compute_region_channel_means_minority(self, benign_images, segmentation_masks):
        """
        benign_images: List[np.ndarray],   (H, W, 3)   
        segmentation_masks: List[torch.Tensor],   (H, W)  segmentation mask 
        
        segmentation mask     (1st minority)     (2nd minority) ,
        benign image      (R, G, B)   tensor .
        
        Returns:
             shape (2, 3) 
                row: [R_mean, G_mean, B_mean] (1st minority class )
                row: [R_mean, G_mean, B_mean] (2nd minority class )
        """
        
        for benign_img, segmentation_mask in zip(benign_images, segmentation_masks):
            #    GPU 
            benign_img = torch.from_numpy(benign_img.copy()).to(self.device)  # (H, W, 3)
            segmentation_mask = segmentation_mask.to(self.device)  # (H, W)
            

            unique_vals, counts = torch.unique(segmentation_mask, return_counts=True)
            
            # 1st 2nd minority  
            sorted_indices = torch.argsort(counts)
            first_minority_class = unique_vals[sorted_indices[0]]
            second_minority_class = unique_vals[sorted_indices[1]] if len(sorted_indices) > 1 else first_minority_class
            

            first_minority_mask = (segmentation_mask == first_minority_class)  # (H, W)
            second_minority_mask = (segmentation_mask == second_minority_class)  # (H, W)
            

            if first_minority_mask.sum() > 0:
                means_first = benign_img[first_minority_mask].float().mean(dim=0)
            else:
                means_first = torch.zeros(3, device=self.device)
                
            if second_minority_mask.sum() > 0:
                means_second = benign_img[second_minority_mask].float().mean(dim=0)
            else:
                means_second = torch.zeros(3, device=self.device)
            

            result_tensor = torch.stack([means_first, means_second])
        
        return result_tensor