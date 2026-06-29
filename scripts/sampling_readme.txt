Segmentation Sample Reproduction Guide
====================================

Purpose
-------
This project reproduces the exact shared sample sets for three datasets:
1. ADE20K
2. Cityscapes
3. VOC2012

The script uses fixed manifest files and copies only listed files from full datasets.

Required Files
--------------
Share these files with third parties:
1. `scripts/extract_ade20k_samples.py`
2. `manifests/ade20k_200_manifest.txt`
3. `manifests/cityscapes_300_manifest.txt`
4. `manifests/voc2012_200_manifest.txt`

Expected Directory Layout
-------------------------
Inside project root:
- `manifests/ade20k_200_manifest.txt`
- `manifests/cityscapes_300_manifest.txt`
- `manifests/voc2012_200_manifest.txt`
- `scripts/extract_ade20k_samples.py`

Fixed source dataset roots:
- `/workspace/Dataset/ADE20K`
- `/workspace/Dataset/cityscapes`
- `/workspace/Dataset/VOCdevkit/VOC2012`

Run
---
From project root:

python scripts/extract_ade20k_samples.py

Output Directories
------------------
After running, reproduced samples are written to:
- `datasets/ade20k`
- `datasets/cityscapes`
- `datasets/VOC2012`

Verification
------------
Check reproduced file counts:

find datasets/ade20k -type f | wc -l
find datasets/cityscapes -type f | wc -l
find datasets/VOC2012 -type f | wc -l
