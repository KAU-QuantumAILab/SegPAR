# SegPAR: Class-Centric Decision-Based Sparse Attack for Semantic Segmentation
[[Paper](#citation)] [[Supp](#supplementary-material)] [[Code](https://github.com/KAU-QuantumAILab/SegPAR)]

The official implementation of **[ECCV 2026] "SegPAR: Class-Centric Decision-Based Sparse Attack for Semantic Segmentation"**.

Paper and supplementary links will be updated after the public release.

## Introduction
SegPAR is a class-centric decision-based sparse attack framework for studying adversarial perturbations on semantic segmentation models. The repository includes attack code, MMSegmentation-based model configurations, dataset sampling manifests, and a Docker runtime for reproducing the experiments.

The current release supports experiments on VOC2012, Cityscapes, ADE20K, and CT Abdomen settings. It includes the `reg` action space used by SegPAR, as well as the `standard` action space used by the baseline setting in `run.sh`.

# Getting Started

## Dependencies
The recommended way to run this repository is with Docker.

The provided `Dockerfile` is based on:

* PyTorch 2.6.0 with CUDA 12.4 and cuDNN 9
* MMEngine 0.10.7
* MMCV 2.2.0
* Python packages including `scikit-image`, `evaluate`, `ftfy`, `regex`, `loralib`, and `setproctitle`

Build the image from the project root:

```bash
docker build -t segpar .
```

## Checkpoints
Download the victim segmentation model checkpoints and place them under the `ckpt/` directory.

All segmentation models used in the experiments are implemented with MMSegmentation and loaded from their corresponding pretrained checkpoints.

Models used in the experiments:

| Dataset | Model | Resolution / Setting |
| --- | --- | --- |
| VOC2012 | DeepLabV3-R101-D8 | 512 x 512, 20k |
| VOC2012 | PSPNet-R101-D8 | 512 x 512, 20k |
| Cityscapes | DeepLabV3-R101-D8 | 512 x 1024, 80k |
| Cityscapes | PSPNet-R101-D8 | 512 x 1024, 80k |
| Cityscapes | SegFormer-MiT-B5 | 1024 x 1024, 160k |
| Cityscapes | SETR-PUP (ViT-Large) | 768 x 768, 80k |
| ADE20K | DeepLabV3-R101-D8 | 512 x 512, 160k |
| ADE20K | PSPNet-R101-D8 | 512 x 512, 160k |
| ADE20K | SegFormer-MiT-B5 | 640 x 640, 160k |
| ADE20K | SETR-PUP (ViT-B/16) | 512 x 512, 160k |

## Datasets
This repository provides fixed manifests for reproducing the shared evaluation samples:

* `manifests/ade20k_200_manifest.txt`
* `manifests/cityscapes_300_manifest.txt`
* `manifests/voc2012_200_manifest.txt`

The sample extraction script expects the full datasets at:

```text
/workspace/Dataset/ADE20K
/workspace/Dataset/cityscapes
/workspace/Dataset/VOCdevkit/VOC2012
```

Run the extraction script from the project root:

```bash
python scripts/extract_ade20k_samples.py
```

The reproduced samples are written to:

```text
datasets/ade20k
datasets/cityscapes
datasets/VOC2012
```

## Running SegPAR
Run the default experiment script inside Docker.

Windows PowerShell:

```powershell
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace segpar bash ./run.sh
```

Linux/macOS:

```bash
docker run --rm --gpus all -v "$(pwd):/workspace" -w /workspace segpar bash ./run.sh
```

The default `run.sh` configuration runs SegPAR on VOC2012 with DeepLabV3:

```bash
python main.py --config configs_attack/VOC2012/config_deeplabv3.py \
  --majority None \
  --bound 100 \
  --patient 1 \
  --process_name deeplabv3_VOC2012_attack_100 \
  --cuda_device cuda:0 \
  --use_gt False \
  --reward_type discrepancy \
  --show_effect True \
  --w 0 \
  --rl_learning_rate 1e-05 \
  --it_max 1000 \
  --action_space reg \
  --batch 4 \
  --attack_pixel 10e-4
```

Set `--action_space reg` for SegPAR. Set `--action_space standard` for the baseline action space used in the script comments.

## Evaluation Configurations
Attack configurations are stored under `configs_attack/`:

```text
configs_attack/VOC2012/
configs_attack/cityscapes/
configs_attack/ade20k/
configs_attack/CT_Abd/
```

Adversarial training and defense-related configurations are stored under `configs_adv/`.

## Results
Experiment outputs are saved under `results/`. The exact output path is generated from the dataset, model, reward type, action space, sparsity level, and other runtime arguments.

## Supplementary Material
This repository was prepared from the SegPAR supplementary material release. Paper and supplementary PDF links will be added here after the ECCV 2026 public release.

## Acknowledgements
This codebase builds on components and configuration styles from [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and includes Segment Anything model utilities.

We thank the authors and maintainers of the open-source projects used in this repository.

## Citation
If you find this repository useful in your research, please cite the SegPAR paper after the official ECCV 2026 proceedings entry is available.

```bibtex
@inproceedings{segpar2026,
  title     = {SegPAR: Class-Centric Decision-Based Sparse Attack for Semantic Segmentation},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

The complete BibTeX entry will be updated after publication.

## Contact
For questions, please open an issue in this repository or contact the KAU-QuantumAILab maintainers.
