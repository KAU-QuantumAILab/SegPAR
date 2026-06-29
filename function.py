import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
import PIL.Image as Image
from typing import List
from torchvision import transforms
import os.path as osp
import cv2
from scipy.ndimage import distance_transform_edt

def data_preprocessing(
    np_image_list: List[np.ndarray],
    transform: transforms.Compose
) -> torch.Tensor:
    """
    Apply torchvision transform to a list of numpy arrays and return as a tensor batch.

    Args:
        np_image_list: List of RGB numpy arrays (each image: (H, W, 3), dtype=np.uint8)
        transform: torchvision.transforms.Compose object

    Returns:
        torch.Tensor: Transformed image tensor batch (shape: [N, C, H, W])
    """
    tensor_list = []
    for np_img in np_image_list:
        if not isinstance(np_img, np.ndarray):
            raise TypeError("The input list should only contain numpy arrays.")
        if np_img.ndim != 3 or np_img.shape[2] != 3:
            raise ValueError("Each image array should be in (H, W, 3) format of RGB image.")
        if np_img.dtype != np.uint8:
            raise ValueError("Image dtype should be np.uint8.")

        pil_img = Image.fromarray(np_img)
        tensor_img = transform(pil_img)
        tensor_list.append(tensor_img)

    return torch.stack(tensor_list)

class StateProcessor:
    def __init__(self, config):
        if config["dataset"] == "cityscapes":
            self.background_id = 255
        elif config["dataset"] == "ade20k":
            self.background_id = 0
        elif config["dataset"] == "VOC2012":
            self.background_id = 0
        elif config["dataset"] == "CT_Abd":
            self.background_id = 255
        else:
            raise ValueError(f"Unsupported dataset: {config['dataset']}")
        self.idx=0
        self.config = config


    def process(self, image, full_mask):
        """
        image: np.array (H, W, 3)
        full_mask: np.array (H, W) or torch.Tensor,    ID  
        """
        #   numpy  
        if isinstance(full_mask, torch.Tensor):
            full_mask = full_mask.cpu().numpy()
        self.idx+=1
        unique_classes = np.unique(full_mask)
        #  ID 
        classes = [c for c in unique_classes if c != self.background_id]
        print(classes)
        if len(classes) == 0:
            # Foreground    no-op   
            #  (torch.stack, batch ) shape mismatch .
            h, w = full_mask.shape[:2]
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return {
                int(self.background_id): {
                    "image": image.copy(),
                    "mask": empty_mask,
                    "active_pixels": 0,
                    "full_mask": full_mask,
                    "bbox": (0, h - 1, 0, w - 1),
                }
            }
        processed_states = {}

        h, w = full_mask.shape[:2]
        total_attack_budget = int(h * w * self.config["attack_pixel"])

        class_masks = {}
        for class_id in classes:
            class_masks[class_id] = (full_mask == class_id).astype(np.uint8)

        # CT_Abd classes 0 ,
        # class 0   bbox   1 
        if self.config["dataset"].lower() == "ct_abd" and 0 in classes:
            bbox_union_mask = np.zeros((h, w), dtype=np.uint8)
            bbox_source_classes = [cid for cid in classes if cid != 0]
            for bbox_class_id in bbox_source_classes:
                bbox_coords = np.argwhere(class_masks[bbox_class_id])
                if bbox_coords.size == 0:
                    continue
                y_min, x_min = bbox_coords.min(axis=0)
                y_max, x_max = bbox_coords.max(axis=0)
                bbox_union_mask[y_min:y_max+1, x_min:x_max+1] = 1

            class_masks[0] = ((full_mask == 0) & (bbox_union_mask == 1)).astype(np.uint8)

        class_areas = {}
        for class_id in classes:
            class_areas[class_id] = int(class_masks[class_id].sum())

        total_class_area = sum(class_areas.values())
        class_active_pixels = {class_id: 0 for class_id in classes}
        if total_class_area > 0 and total_attack_budget > 0:
            raw_allocations = {
                class_id: (total_attack_budget * class_areas[class_id]) / total_class_area
                for class_id in classes
            }
            floor_allocations = {
                class_id: int(np.floor(raw_allocations[class_id]))
                for class_id in classes
            }
            remainder = total_attack_budget - sum(floor_allocations.values())

            if remainder > 0:
                sorted_by_fraction = sorted(
                    classes,
                    key=lambda cid: raw_allocations[cid] - floor_allocations[cid],
                    reverse=True
                )
                for cid in sorted_by_fraction[:remainder]:
                    floor_allocations[cid] += 1

            for class_id in classes:
                class_active_pixels[class_id] = min(
                    class_areas[class_id],
                    floor_allocations[class_id]
                )

        for class_id in classes:

            class_mask = class_masks[class_id]
            active_pixels = class_active_pixels.get(class_id, 0)

            coords = np.argwhere(class_mask)
            if coords.size == 0:
                continue
                
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 3. Bounding Box     
            # (y_max + 1, x_max + 1    )
            cropped_img = image[y_min:y_max+1, x_min:x_max+1]
            cropped_mask = class_mask[y_min:y_max+1, x_min:x_max+1]
            
            # 4.   (     offset   )
            processed_states[class_id] = {
                "image": cropped_img,
                "mask": cropped_mask,
                "active_pixels": active_pixels,
                "full_mask": class_mask,
                "bbox": (y_min, y_max, x_min, x_max)
            }
            
        return processed_states


# processor = StateProcessor(background_id=0)
# states = processor.process(input_np_img, segmentation_mask)
# print(states[1]["image"].shape) #  1 BBox  

def prepare_agent_input(processed_states, config, target_size=(256, 256)):
    """
    StateProcessor     
    """
    batch_images = []
    batch_masks = []
    batch_full_masks = []
    batch_active_pixels = []
    metadata = []

    
    for class_id, data in processed_states.items():
        

        img_res = cv2.resize(data['image'], target_size, interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(data['mask'], target_size, interpolation=cv2.INTER_NEAREST)

        ymin, ymax, xmin, xmax = data['bbox']
        orig_area = (ymax - ymin) * (xmax - xmin)
        attack_pixels = max(1, int(orig_area * config["attack_pixel"]))

        # 2.   (HWC -> CHW)
        if img_res.max()<= 1.0:
            batch_images.append(torch.from_numpy(img_res).permute(2, 0, 1).float())
        else:
            batch_images.append(torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0)
        batch_masks.append(torch.from_numpy(mask_res).unsqueeze(0).float())
        batch_full_masks.append(torch.from_numpy(data['full_mask']).unsqueeze(0).float())
        metadata.append({
            'class_id': class_id,
            'bbox': data['bbox'], # (ymin, ymax, xmin, xmax)
            'orig_shape': data['mask'].shape, # (H, W)
            'active_pixels': data['active_pixels'],
            'attack_pixels': attack_pixels
        })


    return torch.stack(batch_images), torch.stack(batch_masks), torch.stack(batch_full_masks), metadata

class AttackDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, mask_list, config):
        self.processor = StateProcessor(config)
        self.data = []
        self.config = config
        
        for img, mask in zip(image_list, mask_list):
            states = self.processor.process(img, mask)
            processed_input, processed_mask, processed_full_mask, metadata = prepare_agent_input(states, self.config)
            

            self.data.append({
                'input': processed_input,
                'mask': processed_mask,
                'full_mask' : processed_full_mask,
                'meta': metadata
            })
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]

    def delete_attack_pixels(self, idx, prediction_mask):
        data = self.data[idx]
        if isinstance(prediction_mask, torch.Tensor):
            prediction_mask = prediction_mask.cpu().numpy()




        for i, meta_item in enumerate(data['meta']):
            benign_id = meta_item['class_id']
            

            benign_mask = data['full_mask'][i].squeeze()
            if hasattr(benign_mask, 'numpy'): 
                benign_mask = benign_mask.numpy()
            
            current_pred_mask = (prediction_mask == benign_id).astype(np.uint8)
            new_attack_region = current_pred_mask * benign_mask


            if np.argwhere(new_attack_region).size == 0:
                self.data[idx]['meta'][i]['attack_pixels'] = 0
                


def apply_batch_binary_attacks(
    image_list, 
    action_coords_batch, 
    action_values_batch, 
    metadata_batch, 
    masks,
    config,
    target_size=(256, 256)
):
    attacked_images = []
    total_history = []
    start_idx = 0  # (flat)     

    W_ref, H_ref = target_size
    current_data_ptr = 0


    for i in range(len(image_list)):
        original_img = image_list[i]
        perturbed_img = original_img.copy()
        H_orig, W_orig, _ = perturbed_img.shape
        



        #   i  ()  
        current_image_meta = metadata_batch[i]
        img_history = []



        for j in range(len(current_image_meta)):

            
            

            if current_image_meta[j]['attack_pixels'] == 0:
                continue
            
            coords = action_coords_batch[current_data_ptr] # (num_samples, 2)
            rgb_actions = action_values_batch[current_data_ptr] # (num_samples, 3)
            

            if torch.is_tensor(coords): coords = coords.detach().cpu().numpy()
            if torch.is_tensor(rgb_actions): rgb_actions = rgb_actions.detach().cpu().numpy()

            ymin, ymax, xmin, xmax = current_image_meta[j]['bbox']
            bbox_w, bbox_h = xmax - xmin, ymax - ymin

            # 3.   (  num_samples )
            # len(coords)       
            for k in range(len(coords)):
                x_prime, y_prime = coords[k]
                rgb_vals = rgb_actions[k]    #   RGB  

                # ---   (Global Mapping) ---
                orig_x_float = xmin + (float(x_prime) / W_ref) * bbox_w
                orig_y_float = ymin + (float(y_prime) / H_ref) * bbox_h
                
                ix = int(np.clip(round(orig_x_float), 0, W_orig - 1))
                iy = int(np.clip(round(orig_y_float), 0, H_orig - 1))


                if config["attack_type"] == "nearest":
                    # Nearest Neighbor Attack:       
                    current_mask = masks[current_data_ptr]
                    if torch.is_tensor(current_mask):
                        current_mask = current_mask.detach().cpu().numpy()
                    
                    #  (1, 256, 256)  squeeze
                    if current_mask.ndim == 3: current_mask = current_mask.squeeze(0)

                    distances, indices = distance_transform_edt(current_mask, return_indices=True)
                    #  (y', x')      
                    nearest_y_prime, nearest_x_prime = indices[:, int(y_prime), int(x_prime)]
                    

                    target_x = int(np.clip(round(xmin + (float(nearest_x_prime) / W_ref) * bbox_w), 0, W_orig - 1))
                    target_y = int(np.clip(round(ymin + (float(nearest_y_prime) / H_ref) * bbox_h), 0, H_orig - 1))
                    
                    perturbed_img[iy, ix, :] = original_img[target_y, target_x, :]
                elif config["attack_type"] == "standard":
                    # Binary Attack: 0  255 
                    if config["dataset"].lower() == "ct_abd":
                        for c in range(3):
                            perturbed_img[iy, ix, c] = 0 if rgb_vals[c] == 0 else 1

                    else:
                        for c in range(3):
                            perturbed_img[iy, ix, c] = 0 if rgb_vals[c] == 0 else 255



                elif config["attack_type"] == "grey":
                    pixel_val = int(np.sum(rgb_vals)/3)
                    perturbed_img[iy, ix, :] = pixel_val

                


                img_history.append({'coord': (ix, iy), 'current_data_ptr': current_data_ptr})
            current_data_ptr +=1


        attacked_images.append(perturbed_img)
        total_history.append(img_history)
        

    return attacked_images


# Cityscapes original label to trainId mapping dictionary
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
    Convert GT label array (gtFine_labelIds) to trainId used for Cityscapes evaluation.
    Unlabeled labels are processed as ignore index 255.
    """
    ignore_val = 255
    converted = np.full_like(gt_array, ignore_val)
    for orig_label, train_label in CITYSCAPES_LABEL_MAPPING.items():
        converted[gt_array == orig_label] = train_label
    return converted

# Color mapping for Cityscapes classes
CITYSCAPES_COLORMAP = {
    0: (128, 64, 128),    # road
    1: (244, 35, 232),    # sidewalk
    2: (70, 70, 70),      # building
    3: (102, 102, 156),   # wall
    4: (190, 153, 153),   # fence
    5: (153, 153, 153),   # pole
    6: (250, 170, 30),    # traffic light
    7: (220, 220, 0),     # traffic sign
    8: (107, 142, 35),    # vegetation
    9: (152, 251, 152),   # terrain
    10: (70, 130, 180),   # sky
    11: (220, 20, 60),    # person
    12: (255, 0, 0),      # rider
    13: (0, 0, 142),      # car
    14: (0, 0, 70),       # truck
    15: (0, 60, 100),     # bus
    16: (0, 80, 100),     # train
    17: (0, 0, 230),      # motorcycle
    18: (119, 11, 32)     # bicycle
}

ADE20K_COLORMAP = {
    0: (0, 0, 0),            # background / unlabeled
    1: (120, 120, 120),      # wall
    2: (180, 120, 120),      # building
    3: (6, 230, 230),        # sky
    4: (80, 50, 50),         # floor
    5: (4, 200, 3),          # tree
    6: (120, 120, 80),       # ceiling
    7: (140, 140, 140),      # road
    8: (204, 5, 255),        # bed
    9: (230, 230, 230),      # windowpane
    10: (4, 250, 7),
    11: (224, 5, 255),
    12: (235, 255, 7),
    13: (150, 5, 61),
    14: (120, 120, 70),
    15: (8, 255, 51),
    16: (255, 6, 82),
    17: (143, 255, 140),
    18: (204, 255, 4),
    19: (255, 51, 7),
    20: (204, 70, 3),
    21: (0, 102, 200),
    22: (61, 230, 250),
    23: (255, 6, 51),
    24: (11, 102, 255),
    25: (255, 7, 71),
    26: (255, 9, 224),
    27: (9, 7, 230),
    28: (220, 220, 220),
    29: (255, 9, 92),
    30: (112, 9, 255),
    31: (8, 255, 214),
    32: (7, 255, 224),
    33: (255, 184, 6),
    34: (10, 255, 71),
    35: (255, 41, 10),
    36: (7, 255, 255),
    37: (224, 255, 8),
    38: (102, 8, 255),
    39: (255, 61, 6),
    40: (255, 194, 7),
    41: (255, 122, 8),
    42: (0, 255, 20),
    43: (255, 8, 41),
    44: (255, 5, 153),
    45: (6, 51, 255),
    46: (235, 12, 255),
    47: (160, 150, 20),
    48: (0, 163, 255),
    49: (140, 140, 140),
    50: (250, 10, 15),
    51: (20, 255, 0),
    52: (31, 255, 0),
    53: (255, 31, 0),
    54: (255, 224, 0),
    55: (153, 255, 0),
    56: (0, 0, 255),
    57: (255, 71, 0),
    58: (0, 235, 255),
    59: (0, 173, 255),
    60: (31, 0, 255),
    61: (11, 200, 200),
    62: (255, 82, 0),
    63: (0, 255, 245),
    64: (0, 61, 255),
    65: (0, 255, 112),
    66: (0, 255, 133),
    67: (255, 0, 0),
    68: (255, 163, 0),
    69: (255, 102, 0),
    70: (194, 255, 0),
    71: (0, 143, 255),
    72: (51, 255, 0),
    73: (0, 82, 255),
    74: (0, 255, 41),
    75: (0, 255, 173),
    76: (10, 0, 255),
    77: (173, 255, 0),
    78: (0, 255, 153),
    79: (255, 92, 0),
    80: (255, 0, 255),
    81: (255, 0, 245),
    82: (255, 0, 102),
    83: (255, 173, 0),
    84: (255, 0, 20),
    85: (255, 184, 184),
    86: (0, 31, 255),
    87: (0, 255, 61),
    88: (0, 71, 255),
    89: (255, 0, 204),
    90: (0, 255, 194),
    91: (0, 255, 82),
    92: (0, 10, 255),
    93: (0, 112, 255),
    94: (51, 0, 255),
    95: (0, 194, 255),
    96: (0, 122, 255),
    97: (0, 255, 163),
    98: (255, 153, 0),
    99: (0, 255, 10),
    100: (255, 112, 0),
    101: (143, 255, 0),
    102: (82, 0, 255),
    103: (163, 255, 0),
    104: (255, 235, 0),
    105: (8, 184, 170),
    106: (133, 0, 255),
    107: (0, 255, 92),
    108: (184, 0, 255),
    109: (255, 0, 31),
    110: (0, 184, 255),
    111: (0, 214, 255),
    112: (255, 0, 112),
    113: (92, 255, 0),
    114: (0, 224, 255),
    115: (112, 224, 255),
    116: (70, 184, 160),
    117: (163, 0, 255),
    118: (153, 0, 255),
    119: (71, 255, 0),
    120: (255, 0, 163),
    121: (255, 204, 0),
    122: (255, 0, 143),
    123: (0, 255, 235),
    124: (133, 255, 0),
    125: (255, 0, 235),
    126: (245, 0, 255),
    127: (255, 0, 122),
    128: (255, 245, 0),
    129: (10, 190, 212),
    130: (214, 255, 0),
    131: (0, 204, 255),
    132: (20, 0, 255),
    133: (255, 255, 0),
    134: (0, 153, 255),
    135: (0, 41, 255),
    136: (0, 255, 204),
    137: (41, 0, 255),
    138: (41, 255, 0),
    139: (173, 0, 255),
    140: (0, 245, 255),
    141: (71, 0, 255),
    142: (122, 0, 255),
    143: (0, 255, 184),
    144: (0, 92, 255),
    145: (184, 255, 0),
    146: (0, 133, 255),
    147: (255, 214, 0),
    148: (25, 194, 194),
    149: (102, 255, 0),
    150: (92, 0, 255)
}

VOC2012_COLORMAP = {0: (0, 0, 0),
1: (128, 0, 0),
2: (0, 128, 0),
3: (128, 128, 0),
4: (0, 0, 128),
5: (128, 0, 128),
6: (0, 128, 128),
7: (128, 128, 128),
8: (64, 0, 0),
9: (192, 0, 0),
10: (64, 128, 0),
11: (192, 128, 0),
12: (64, 0, 128),
13: (192, 0, 128),
14: (64, 128, 128),
15: (192, 128, 128),
16: (0, 64, 0),
17: (128, 64, 0),
18: (0, 192, 0),
19: (128, 192, 0),
20: (0, 64, 128)
}

# FLARE22 CT_Abd label colormap (0: background, 1~13 organs)
CT_ABD_COLORMAP = {
    0: (0, 0, 0),         # background
    1: (255, 99, 71),
    2: (30, 144, 255),
    3: (50, 205, 50),
    4: (255, 215, 0),
    5: (186, 85, 211),
    6: (255, 140, 0),
    7: (0, 206, 209),
    8: (220, 20, 60),
    9: (106, 90, 205),
    10: (0, 191, 255),
    11: (127, 255, 0),
    12: (255, 20, 147),
    13: (210, 180, 140),
}

def visualize_segmentation(
    image:     np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    alpha:     float = 0.5,
    dataset:   str   = "cityscapes"
) -> None:
    """
    Create a color overlay of the segmentation mask on the input image
    *and* save the colored mask itself as a separate PNG.

    Args
    image     : np.ndarray (H, W, 3, BGR)  
        Original input image in **BGR** order (as loaded by OpenCV).

    pred_mask : np.ndarray (H, W)  
        Predicted mask where each pixel stores a **class index**.

    save_path : str  
        Destination path for the overlay PNG ('.png' extension optional).

    alpha     : float (0‒1)  
        Transparency of the mask overlay (0 = invisible, 1 = fully opaque).

    dataset   : str  
        Which dataset colormap to use:
        'cityscapes' | 'ade20k' | 'voc2012' | 'ct_abd' (or 'CT_Abd').

    Returns
    None (images are written to disk).
    """

    # 1. BGR → RGB conversion + dtype normalization
    #    cv2.addWeighted  dtype  .

    image_rgb = image[..., ::-1]
    if image_rgb.dtype != np.uint8:
        # float [0,1]      
        if np.issubdtype(image_rgb.dtype, np.floating):
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255.0).round()
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)


    # 2. Retrieve the appropriate colormap dictionary

    dataset_key = dataset.lower()
    if dataset_key   == "cityscapes":
        mapping = CITYSCAPES_COLORMAP
    elif dataset_key == "ade20k":
        mapping = ADE20K_COLORMAP
    elif dataset_key == "voc2012":
        mapping = VOC2012_COLORMAP
    elif dataset_key in ["ct_abd", "ctabd", "ct-abd"]:
        mapping = CT_ABD_COLORMAP
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")


    # 3. Build a (N_classes × 3) lookup table once, then index
    #    into it to create an RGB mask in a vectorized fashion.

    max_idx  = max(mapping.keys())
    colormap = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for idx, color in mapping.items():
        colormap[idx] = color

    pred_mask = np.asarray(pred_mask).astype(np.int64)
    pred_mask_safe = np.where(
        (pred_mask >= 0) & (pred_mask <= max_idx),
        pred_mask,
        0,
    )
    colored_mask = colormap[pred_mask_safe]   # (H, W, 3), RGB


    # 4. Blend mask and image to create the overlay

    overlay = cv2.addWeighted(image_rgb, 1 - alpha,
                              colored_mask, alpha, 0)


    # 5. Ensure the output directory exists

    save_dir = osp.dirname(save_path)
    if save_dir and not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    # 6. Force '.png' extension if it was omitted

    if not save_path.lower().endswith(".png"):
        save_path += ".png"


    # 7. Save overlay (RGB) as PNG

    Image.fromarray(overlay).save(save_path,
                                  format="PNG",
                                  optimize=True)


    # 8. Save colored mask with a 'pred_' prefix

    base_name  = osp.basename(save_path)          # e.g. 'result.png'
    colored_nm = f"pred_{base_name}"              # e.g. 'pred_result.png'
    colored_fp = osp.join(save_dir, colored_nm)

    Image.fromarray(colored_mask).save(colored_fp,
                                       format="PNG",
                                       optimize=True)





#   Kaiming  

def kaiming_init_he(model: nn.Module,
                    only_trainable: bool = True,
                    include_conv: bool = False) -> None:
    """
     nn.Linear (  Conv2d ) 
    He(=Kaiming) Normal  .

    Args
    model :   torch.nn.Module
    only_trainable : True requires_grad=True  
    include_conv   : True Conv2d   
    """
    for m in model.modules():
        is_linear = isinstance(m, nn.Linear)
        is_conv   = include_conv and isinstance(m, nn.Conv2d)

        if is_linear or is_conv:
            if only_trainable and not m.weight.requires_grad:
                continue
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)


    # policy = REINFORCE_Policy(config).to(config["device"])

    # #   (freeze)   
    # kaiming_init_he(policy, only_trainable=True, include_conv=False)
