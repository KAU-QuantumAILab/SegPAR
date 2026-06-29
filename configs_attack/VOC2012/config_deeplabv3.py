# config.py

config = {
    "task": "segmentation",
    "dataset": "VOC2012",
    "data_dir": "datasets/VOC2012",         # Directory path where the dataset is located
    "base_dir": "results/VOC2012/deeplabv3",  # Base directory for saving results
    "model": "deeplabv3",
    "action_dim": 5,                                # Action dimension (adjust as needed)
    "RGB": 3,                                       # Input dimension
    "RL_learning_rate": 0.00001,                    # RL learning rate (e.g., CIFAR10: 0.0001, ImageNet: 0.00005)
    "bound": 100,                                   # Number of iterations for the Forget process
    "limit": 1e-2,                                  # Convergence criterion
    "attack_pixel": 10e-4,                              # Attack dimension for the Remember process (recalculated later)
    "patient": 1,                                   # Patience for early stopping
    "img_size_x": 224,                              # Input width for RL
    "img_size_y": 224,                              # Input height for RL
    "num_class": 21,
    "batch": 4,
    "baseline": 5,
    "background": 0,
    "majority": None
}
