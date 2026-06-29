import os
import re
import json
from typing import Dict, List, Tuple, Optional


def normalize_path(path: str) -> str:
    """  (   results   )"""
    #   
    normalized = path.replace("\\", "/")
    # results   
    if "results" in normalized:
        normalized = "results" + normalized.split("results", 1)[1]
    return normalized


def create_empty_results() -> Dict[str, List[float]]:
    """   """
    return {
        "iteration_mean": [],
        "ratio_mean": [],
        "adv_mean_iou": [],
        "adv_mean_accuracy": []
    }

def collect_from_folder(target_folder: str, bound: int, results_list: Dict[str, List[float]], 
                       benign_bound: int = 20) -> None:
    """   """
    if not os.path.isdir(target_folder):
        return

    file_names = [
        fn for fn in os.listdir(target_folder)
        if fn.startswith("experiment_results") and fn.endswith(".txt")
    ]
    file_names.sort()

    # benign_bound  benign   
    if bound == benign_bound and file_names:
        first_file_path = os.path.join(target_folder, file_names[0])
        with open(first_file_path, "r", encoding="utf-8") as f:
            first_text = f.read()

        benign_miou_match = re.search(
            r'benign_miou_score\s*:\s*\{[^}]*"mean_iou"\s*:\s*([0-9eE+\.-]+)',
            first_text,
        )
        benign_macc_match = re.search(
            r'benign_miou_score\s*:\s*\{[^}]*"mean_accuracy"\s*:\s*([0-9eE+\.-]+)',
            first_text,
        )
        if benign_miou_match and benign_macc_match:
            results_list["iteration_mean"].append(0.0)
            results_list["ratio_mean"].append(0.0)
            results_list["adv_mean_iou"].append(float(benign_miou_match.group(1)))
            results_list["adv_mean_accuracy"].append(float(benign_macc_match.group(1)))

    for file_name in file_names:
        file_path = os.path.join(target_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        #   
        iteration_match = re.search(r"iteration_mean\s*[:=]\s*([\d\.eE+-]+)", text)
        ratio_match = re.search(r"ratio_mean\s*[:=]\s*([\d\.eE+-]+)", text)
        #   adv_miou_score: { "mean_iou": 0.123, "mean_accuracy": 0.456, ... } 
        miou_match = re.search(r'adv_miou_score\s*:\s*\{[^}]*"mean_iou"\s*:\s*([0-9eE+\.-]+)', text)
        macc_match = re.search(r'adv_miou_score\s*:\s*\{[^}]*"mean_accuracy"\s*:\s*([0-9eE+\.-]+)', text)

        if iteration_match and ratio_match and miou_match and macc_match:
            results_list["iteration_mean"].append(float(iteration_match.group(1)))
            results_list["ratio_mean"].append(float(ratio_match.group(1)))
            results_list["adv_mean_iou"].append(float(miou_match.group(1)))
            results_list["adv_mean_accuracy"].append(float(macc_match.group(1)))


def collect_results(base_dir: str, bound_range: Tuple[int, int, int] = (20, 101, 20), 
                   final_bound: int = 100, benign_bound: int = 20) -> Dict[str, List[float]]:
    """  
    
    Args:
        base_dir:   
        bound_range: (, +1, ) 
        final_bound:  bound  (base_dir  )
        benign_bound: benign   bound 
    
    Returns:
          
    """
    results_list = create_empty_results()
    
    # bound   (float int )
    start, stop, step = bound_range
    for bound in range(int(start), int(stop), int(step)):
        if bound == final_bound:
            # final_bound  base_dir   
            collect_from_folder(base_dir, bound, results_list, benign_bound)
        else:
            collect_from_folder(os.path.join(base_dir, f"intermediate_bound_{bound}"), 
                              bound, results_list, benign_bound)
    
    return results_list


def save_results_to_json(results: Dict[str, List[float]], output_path: str) -> None:
    """ JSON  """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def export_experiment_results(base_dir_raw: str, bound_range: Tuple[int, int, int] = (20, 101, 20),
                            final_bound: int = 100, benign_bound: int = 20, 
                            print_results: bool = True) -> Dict[str, List[float]]:
    """    
    
    Args:
        base_dir_raw:    
        bound_range: (, +1, ) 
        final_bound:  bound 
        benign_bound: benign   bound 
        print_results:   
    
    Returns:
          
    """
    #  
    base_dir = normalize_path(base_dir_raw)
    
    #  
    results = collect_results(base_dir, bound_range, final_bound, benign_bound)
    
    # 
    if print_results:
        print("Collected Results:")
        for k, v in results.items():
            print(f"{k}: {v}")
    
    # JSON 
    output_path = os.path.join(base_dir, "result.json")
    save_results_to_json(results, output_path)
    
    if print_results:
        print(f"Saved JSON: {output_path}")
    
    return results


