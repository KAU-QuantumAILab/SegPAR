import json
import matplotlib.pyplot as plt
import os

def plot_adversarial_results(dataset_name):
    """
      JSON        .
    """
    file_path = f'results_{dataset_name}.json'
    print(file_path)
    
    #    
    if not os.path.exists(file_path):
        print(f": {file_path}    . .")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    #    
    colors = {
        'standard': 'blue', 
        'discrepancy': 'green', 
        'reduction': 'red', 
        'pixle': 'purple', 
        'sparse-rs': 'orange',
        'pixle_discrepancy': 'purple', 
        'sparse-rs_discrepancy': 'orange'
    }

    for model_name, methods in data.items():
        plt.figure(figsize=(10, 6))
        
        for method_name, values in methods.items():
            x = values["iteration_mean"]
            base_iou = values["adv_mean_iou"][0]
            
            #   (Drop)
            y = [base_iou - val for val in values["adv_mean_iou"]]
            sizes = [r * 2000 for r in values["ratio_mean"]] 
            
            plt.plot(x, y, label=method_name, color=colors.get(method_name.lower(), 'black'), alpha=0.5)
            plt.scatter(x, y, s=sizes, color=colors.get(method_name.lower(), 'black'), edgecolors='black', alpha=0.7)

        #   (  )
        plt.title(f"Attack Performance Drop ({model_name.upper()}) - {dataset_name.upper()}", fontsize=15)
        plt.xlabel("Iteration Mean", fontsize=12)
        plt.ylabel("mIoU Drop ($\Delta$ mIoU)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        plt.text(0.95, 0.05, f"Marker size: pixel ratio", 
                 transform=plt.gca().transAxes, fontsize=10, style='italic', horizontalalignment='right')

        plt.tight_layout()

        #   (    )
        save_name = f'attack_results_{dataset_name}_{model_name}.png'
        plt.savefig(save_name, dpi=600)
        print(f" : {save_name}")
        plt.close()

# ==========================================
# :       .
# ==========================================
datasets = ['voc_decision'] # ['voc', 'ade', 'cityscapes', 'ade_decision', 'voc_decision']

for ds in datasets:
    plot_adversarial_results(ds)