# config.py

config = {
    "task": "segmentation",
    "dataset": "cityscapes",
    "data_dir": "datasets/cityscapes",         # Directory path where the dataset is located
    "base_dir": "results/cityscapes/deeplabv3_ddcat",  # Base directory for saving results
    "action_dim": 5,                                # Action dimension (adjust as needed)
    "RGB": 3,                                       # Input dimension
    "RL_learning_rate": 0.00001,                    # RL learning rate (e.g., CIFAR10: 0.0001, ImageNet: 0.00005)
    "bound": 100,                                   # Number of iterations for the Forget process
    "limit": 1e-2,                                  # Convergence criterion
    "attack_pixel": 0.0001,                              # Attack dimension for the Remember process (recalculated later)
    "patient": 1,                                   # Patience for early stopping
    "img_size_x": 224,                              # Input width for RL
    "img_size_y": 224,                              # Input height for RL
    "num_class": 19,
    "batch": 4,
    "baseline": 5,
    "background": 255,
    "majority": None,

    ######### adv model setting #########
    "model": "deeplabv3_ddcat",
    "model_path": "adv_models/pretrain/cityscapes/deeplabv3/ddcat/train_epoch_400.pth",
    "layers": 50,
    "zoom_factor": 8,
    "scales": [1.0],
    "base_size": 1024,
    "crop_h": 449,
    "crop_w": 449,
    "stride_rate": 2/3,
    "process_name": "deeplabv3_ddcat_attack",
    "use_gt": False,
    "mean": [255*0.485, 255*0.456, 255*0.406],  # [R, G, B]
    "std": [255*0.229, 255*0.224, 255*0.225]    # [R, G, B]
}
