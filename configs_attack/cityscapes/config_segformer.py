# config.py

config = {
    "task": "segmentation",
    "dataset": "cityscapes",
    "data_dir": "datasets/cityscapes",         # Directory path where the dataset is located
    "base_dir": "results/cityscapes/segformer",  # Base directory for saving results
    "model": "segformer",
    "tile_size" : 1024,
    "action_dim": 5,                                # Action dimension (adjust as needed)
    "RGB": 3,                                       # Input dimension
    "RL_learning_rate": 0.00001,                    # RL learning rate (e.g., CIFAR10: 0.0001, ImageNet: 0.00005)
    "gt_dir": "datasets/cityscapes/gtFine",           # Directory for Ground-Truth images
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
    "use_gt": False,
    "original_mask": False
}
