SegPAR Guide
===============

1. Save extracted dataset samples
---------------------------------
Read the sample extraction guide in `scripts/sampling_readme.txt` and prepare the datasets.
ADE20K and VOC2012 sample data are already included in this project.

Run from the project root:

python scripts/extract_ade20k_samples.py

The extracted samples are saved under:
- `datasets/ade20k`
- `datasets/cityscapes`
- `datasets/VOC2012`

2. Save model checkpoints in `ckpt`
-----------------------------------
Download the victim model checkpoints and place them in the `ckpt` directory.
All segmentation models used in our experiments were implemented with MMSegmentation
and loaded from the corresponding pretrained checkpoints from that platform.

Example location:
- `ckpt/...`

Models used in our experiments:

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

3. Build and run with Docker
----------------------------
Build the Docker image from the project root:

docker build -t segpar .

Run the project with the current project directory mounted into the container:

Windows PowerShell:
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace segpar bash ./run.sh

Linux/macOS shell:
docker run --rm --gpus all -v "$(pwd):/workspace" -w /workspace segpar bash ./run.sh

Replace the host path only if you want to mount a different local project directory.

Note
----
The current Dockerfile covers the runtime dependencies needed to start `./run.sh`.
If execution still fails after startup, the remaining issue is likely in project code or config rather than missing container packages.
