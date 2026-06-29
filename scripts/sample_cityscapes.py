import os
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# 설정 파일에서 경로 가져오기
config = {
    "data_dir": "../../Dataset/cityscapes",
    "num_samples": 100
}

def sample_cityscapes():
    # torchvision의 Cityscapes 데이터셋 사용
    from torchvision.datasets import Cityscapes
    
    # Cityscapes 데이터셋 로드
    dataset = Cityscapes(
        root=config['data_dir'],
        split='val',
        mode='fine',
        target_type='semantic'
    )
    
    # 결과 저장 디렉토리 설정 (중복 경로 제거)
    output_img_dir = os.path.join('leftImg8bit', 'val')
    output_gt_dir = os.path.join('gtFine', 'val')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_gt_dir, exist_ok=True)
    
    # 데이터셋에서 이미지와 레이블 가져오기
    all_images = []
    for i in range(len(dataset)):
        image_path = dataset.images[i]
        city = os.path.basename(os.path.dirname(image_path))
        img_name = os.path.basename(image_path)
        all_images.append((city, img_name))
    
    # 랜덤 샘플링
    sampled_images = random.sample(all_images, min(config['num_samples'], len(all_images)))
    
    # 샘플링된 이미지와 GT 복사
    for city, img_name in sampled_images:
        # 원본 이미지 경로
        img_path = os.path.join(config['data_dir'], 'leftImg8bit', 'val', city, img_name)
        
        # GT 이미지 경로
        gt_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
        gt_path = os.path.join(config['data_dir'], 'gtFine', 'val', city, gt_name)
        
        # 도시별 디렉토리 생성
        city_img_dir = os.path.join(output_img_dir, city)
        city_gt_dir = os.path.join(output_gt_dir, city)
        os.makedirs(city_img_dir, exist_ok=True)
        os.makedirs(city_gt_dir, exist_ok=True)
        
        # 결과 저장
        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(city_img_dir, img_name))
        if os.path.exists(gt_path):
            shutil.copy2(gt_path, os.path.join(city_gt_dir, gt_name))
    
    print(f"총 {len(sampled_images)}개의 이미지와 GT가 샘플링되어 저장되었습니다.")
    print(f"이미지 저장 경로: {os.path.abspath(output_img_dir)}")
    print(f"GT 저장 경로: {os.path.abspath(output_gt_dir)}")

if __name__ == '__main__':
    sample_cityscapes() 