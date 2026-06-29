# config.py

config = {
    "task": "segmentation",
    "dataset": "VOC2012",
    "data_dir": "datasets/VOC2012",         # Directory path where the dataset is located
    "base_dir": "results/VOC2012/pspnet_ddcat",  # Base directory for saving results
    "action_dim": 5,                                # Action dimension (adjust as needed)
    "RGB": 3,                                       # Input dimension
    "RL_learning_rate": 0.00001,                    # RL learning rate (e.g., CIFAR10: 0.0001, ImageNet: 0.00005)
    "bound": 100,                                   # Number of iterations for the Forget process
    "limit": 1e-2,                                  # Convergence criterion
    "attack_pixel": 0.0001,                              # Attack dimension for the Remember process (recalculated later)
    "patient": 1,                                   # Patience for early stopping
    "img_size_x": 224,                              # Input width for RL
    "img_size_y": 224,                              # Input height for RL
    "num_class": 21,
    "batch": 4,
    "baseline": 5,
    "background": 0,
    "majority": None,

    ######### adv model setting #########
    "model": "pspnet_ddcat",
    "model_path": "adv_models/pretrain/voc2012/pspnet/ddcat/train_epoch_50.pth",
    "layers": 50,
    "zoom_factor": 8,
    "scales": [1.0],
    "base_size": 512,
    "crop_h": 473,
    "crop_w": 473,
    "stride_rate": 2/3,
    "process_name": "pspnet_ddcat_attack",
    "use_gt": False,
    "mean": [255*0.485, 255*0.456, 255*0.406],  # [R, G, B]
    "std": [255*0.229, 255*0.224, 255*0.225]    # [R, G, B]
}
