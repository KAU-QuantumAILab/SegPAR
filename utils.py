import random
import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
import os
from PIL import  Image
from torch.utils.data import Dataset
import json
import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

def seed_all(seed):
    #  Set fixed seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True









def sample_action(actions_mean, action_std, attack_pixels):
    #  Function to sample actions
    if not isinstance(action_std, torch.Tensor):
        cov_mat = torch.eye(attack_pixels) * action_std**2
    else:
        cov_mat = torch.diag_embed(action_std.pow(2))
    distribution = dist.MultivariateNormal(actions_mean, cov_mat)
    if attack_pixels == 1:
        actions = distribution.sample()
    else:
        actions = distribution.sample_n(attack_pixels)
    

    actions_logprob = distribution.log_prob(actions)


    return actions, actions_logprob







def early_stopping(metric_value: float, patience_counter: int, 
                  min_improvement: float = 10.0, max_patience: int = 5) -> Tuple[int, bool]:
    """
     (early stopping)   .
    
    Args:
        metric_value (float):      (: loss, accuracy)
        patience_counter (int):      
        min_improvement (float):      (: 10.0)
        max_patience (int):         (: 5)
                           0       
        
    Returns:
        Tuple[int, bool]: 
            -  patience_counter
            -    (True  )
    """
    should_stop = False
    
    # max_patience 0  
    if max_patience == 0:
        should_stop = True
        return patience_counter, should_stop
    
    #     
    if metric_value <= min_improvement:
        patience_counter += 1
        
        #      
        if patience_counter >= max_patience:
            should_stop = True
    else:
        #     
        patience_counter = 0
    
    return patience_counter, should_stop



def update(inputs1, inputs2, indices, all_same_shape):
    """
              update .
    all_same_shape True  GPU   .
    
    Args:
        np_input1: NumPy   NumPy  
        np_input2: NumPy   NumPy  
        np_indices: NumPy   NumPy   ( 0 np_input1  )
        all_same_shape:     
    
    Returns:
        NumPy   NumPy  :  
    """

    if all_same_shape:
        if isinstance(inputs1, list):
            result_list = [
                inputs1[i] if indices[i] == 0 else inputs2[i]
                for i in range(indices.shape[0])
            ]
            return result_list
        #  
        elif isinstance(inputs1, torch.Tensor):
            if inputs1.ndim == 4 or inputs1.ndim == 3:
                results = []
                batch_size = 4  #  
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for i in range(0, indices.shape[0], batch_size):
                    end_idx = min(i + batch_size, indices.shape[0])
                    if inputs1.ndim == 4:
                        batch_indices = indices[i:end_idx].view(-1, 1, 1, 1).to(device)
                    else:  # ndim == 3
                        batch_indices = indices[i:end_idx].view(-1, 1, 1).to(device)
                    batch_input1 = inputs1[i:end_idx].to(device)
                    batch_input2 = inputs2[i:end_idx].to(device)
                    batch_result = torch.where(batch_indices == 0, batch_input1, batch_input2)
                    results.append(batch_result.cpu())
                return torch.cat(results, dim=0)
            else:
                return torch.where(indices == 0, inputs1, inputs2).cpu()
    else:
        results = []

        if isinstance(inputs1, torch.Tensor) & isinstance(inputs2, torch.Tensor):
            results = torch.where(indices == 0, inputs1, inputs2)
        else:
            for i in range(len(inputs1)):
                #     
                input1 = inputs1[i]
                input2 = inputs2[i]
                idx = indices[i]

                
                #   input1  input2 
                if isinstance(input1, torch.Tensor):
                    result = torch.where(idx == 0, input1, input2)
                else:
                    result = np.where(idx == 0, input1, input2)
                results.append(result)
            
        return results
    




# def update(np_input1, np_input2, np_indices, all_same_shape):
#     """
#               update .
#     all_same_shape True  GPU   .
    
#     Args:
#         np_input1: NumPy   NumPy  
#         np_input2: NumPy   NumPy  
#         np_indices: NumPy   NumPy   ( 0 np_input1  )
#         all_same_shape:     
    
#     Returns:
#         NumPy   NumPy  :  
#     """

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     #   
#     is_list_input = isinstance(np_input1, list)
#     is_numpy_input = isinstance(np_input1, np.ndarray)
#     is_tensor_input = isinstance(np_input1, torch.Tensor)
    
#     if is_numpy_input:
#         return np.where(np_indices == 0, np_input1, np_input2)
    
#     if all_same_shape and is_tensor_input:
#         # GPU  
#         #   
#         if isinstance(np_input2, torch.Tensor):
#             input2_tensor = np_input2
#         else:
#             input2_tensor = torch.from_numpy(np.stack(np_input2)).to(device)
        
#         # indices  shape  (  )
#         if np_indices[0].ndim == 0:  #  
#             indices_tensor = torch.tensor(np_indices, device=device).view(-1, *([1] * (np_input1.ndim - 1)))
#         else:
#             #     
#             indices_tensor = torch.from_numpy(np.stack([
#                 np.broadcast_to(idx, np_input1.shape[1:])
#                 for idx in np_indices
#             ])).to(device)
        
#         #   
#         result_tensor = torch.where(indices_tensor == 0, np_input1, input2_tensor)
        
#         # NumPy    
#         return result_tensor
    
#     else:
#         #  :  
#         results = []
        
#         for i in range(len(np_input1)):
#             #     
#             input1 = np_input1[i]
#             input2 = np_input2[i]
#             idx = np_indices[i]

            
#             #   input1  input2 
#             if isinstance(input1, torch.Tensor) & isinstance(input2, torch.Tensor):
#                 result = torch.where(torch.tensor(idx).to(input1.device) == 0, input1, input2)
#             elif isinstance(input2, torch.Tensor):
#                 result = np.where(idx == 0, input1, input2.cpu().numpy())
#             elif isinstance(input1, torch.Tensor):
#                 result = np.where(idx == 0, input1.cpu().numpy(), input2)
#             else:
#                 result = np.where(idx == 0, input1, input2)
#             results.append(result)
        
#         return results


#Visualization

# Cityscapes 19     (  [R, G, B])
cityscapes_palette = np.array([
    [128, 64,128],   # 0: road
    [244, 35,232],   # 1: sidewalk
    [ 70, 70, 70],   # 2: building
    [102,102,156],   # 3: wall
    [190,153,153],   # 4: fence
    [153,153,153],   # 5: pole
    [250,170, 30],   # 6: traffic light
    [220,220,  0],   # 7: traffic sign
    [107,142, 35],   # 8: vegetation
    [152,251,152],   # 9: terrain
    [ 70,130,180],   # 10: sky
    [220, 20, 60],   # 11: person
    [255,  0,  0],   # 12: rider
    [ 0,  0,142],    # 13: car
    [ 0,  0, 70],    # 14: truck
    [ 0, 60,100],    # 15: bus
    [ 0, 80,100],    # 16: train
    [ 0,  0,230],    # 17: motorcycle
    [119, 11, 32]    # 18: bicycle
])


def overlay_mask_on_image(image, mask, alpha=0.3):
    """
       segmentation mask alpha blending   .
    """
    #  torch.Tensor numpy array 
    if isinstance(image, torch.Tensor):
        image_np = image.cpu().numpy()
    else:
        image_np = np.array(image)
    
    #   dtype    float   blending
    overlay = image_np.astype(np.float32) * (1 - alpha) + mask.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay    


















class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # NumPy   
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()  #   
        elif isinstance(obj, torch.device):
            return str(obj)  # device   
        return super().default(obj)

def save_experiment_results(results, config, sweep_config=None, timestamp=None, save_dir="."):
    """
    config sweep_config,       .
    
    Args:
        results (dict):     (: {"accuracy": 0.85, "loss": 0.35, ...})
        config (dict):   (: config.py  config )
        sweep_config (dict, optional):    (: config.py  sweep_config )
        save_dir (str):     (:  )
        
     "experiment_results.txt"  .
    """
    #      
    if timestamp == None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"experiment_results_{sweep_config}.txt"
    file_path = os.path.join(save_dir, file_name)
    
    lines = []
    lines.append("Experiment Results")
    lines.append("=" * 40)
    lines.append("Timestamp: " + str(datetime.datetime.now()))
    lines.append("\n[Configuration]")
    lines.append(json.dumps(config, indent=4, ensure_ascii=False, cls=CustomJSONEncoder))
    
    if sweep_config is not None:
        lines.append("\n[Sweep Configuration]")
        lines.append(json.dumps(sweep_config, indent=4, ensure_ascii=False, cls=CustomJSONEncoder))
    
    # Experimental Results  (    key: value  )
    lines.append("\n[Experimental Results]")
    for key, value in results.items():
        line = f"{key}: " + json.dumps(value, ensure_ascii=False, cls=CustomJSONEncoder)
        lines.append(line)
    
    os.makedirs(save_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Results saved to {file_path}")

def convert_to_train_id(label_array):
    """
    Cityscapes      .
    """
    mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18
    }
    return np.vectorize(lambda x: mapping.get(x, 255))(label_array)