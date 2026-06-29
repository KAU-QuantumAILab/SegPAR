import os
import numpy as np
from PIL import Image

class CitySet:
    def __init__(self, dataset_dir=None, images=None, gt_images=None, use_gt=False):
        """
        Load and convert Cityscapes dataset images to NumPy arrays.
        
        Args:
            dataset_dir (str, optional): Path to Cityscapes dataset directory
                             (e.g. "Projects/mmsegmentation/datasets/cityscapes")
                             
                             Cityscapes dataset directory structure:
                             cityscapes/
                             ├── leftImg8bit/
                             │   ├── train/
                             │   │   ├── aachen/
                             │   │   │   ├── aachen_000000_000019_leftImg8bit.png
                             │   │   ├── bochum/
                             │   ├── val/
                             │   └── test/
                             └── gtFine/
                                 ├── train/
                                 │   ├── aachen/
                                 │   │   ├── aachen_000000_000019_gtFine_labelIds.png
                             │   ├── bochum/
                             ├── val/
                             └── test/
                                 
            images (list, optional): List of images as NumPy arrays
        """
        self.use_gt = use_gt
        self.benign_pred = None
        if dataset_dir is not None:
            # Verify Cityscapes dataset structure
            leftimg_dir = os.path.join(dataset_dir, "leftImg8bit")
            if not os.path.exists(leftimg_dir):
                raise ValueError(f"Invalid Cityscapes dataset structure. {leftimg_dir} does not exist.")
                
            self.dataset_dir = os.path.join(dataset_dir, "leftImg8bit", "val")
            if not os.path.exists(self.dataset_dir):
                raise ValueError(f"Validation dataset not found at: {self.dataset_dir}")
                
            self.gt_dir = os.path.join(dataset_dir, "gtFine", "val")
            if use_gt and not os.path.exists(self.gt_dir):
                raise ValueError(f"Ground truth directory not found at: {self.gt_dir}")
                
            self.images, self.filenames, self.gt_images = self._load_images()
        elif images is not None:
            self.images = images
            # Generate filenames as image_0.png, image_1.png, etc.
            self.filenames = [f"image_{i}.png" for i in range(len(images))]
            # For images provided directly, we don't have ground truth
            if gt_images is not None:
                self.gt_images = gt_images

        else:
            raise ValueError("Either dataset_dir or images must be provided")
    
    def _load_images(self):
        """
        Load all images from Cityscapes dataset and convert to list of NumPy arrays.
        Images are converted to BGR format.
        
        
        Returns:
            tuple: (list of images as NumPy arrays in BGR format, list of corresponding filenames, list of ground truth images)
        """
        images = []
        filenames = []
        gt_images = []
        
        # Iterate through city directories in Cityscapes
        for city in os.listdir(self.dataset_dir):
            city_path = os.path.join(self.dataset_dir, city)
            
            if os.path.isdir(city_path):
                # Only load Cityscapes format images (ending with _leftImg8bit.png)
                for filename in os.listdir(city_path):
                    if filename.endswith("_leftImg8bit.png"):
                        file_path = os.path.join(city_path, filename)
                        
                        try:
                            # Load image and convert to NumPy array
                            img = Image.open(file_path)
                            img_array = np.array(img)
                            
                            # RGB -> BGR conversion (Cityscapes stores in RGB)
                            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                                img_array = img_array[:, :, ::-1]
                            
                            images.append(img_array)
                            filenames.append(os.path.join(city, filename))  # Store as city/filename
                            


                            gt_filename = filename.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                            gt_path = os.path.join(self.gt_dir, city, gt_filename)
                            gt_img = Image.open(gt_path)
                            gt_array = np.array(gt_img)
                            gt_images.append(gt_array)

                        except Exception as e:
                            print(f"Error loading image ({city}/{filename}): {e}")
        
        if not images:
            print("Warning: No images loaded. Please check dataset path.")
            
        return images, filenames, gt_images
    
    def __len__(self):
        """
        Return the number of images in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Return the image, filename and ground truth at the specified index.
        
        Args:
            idx (int): Dataset index
            
        Returns:
            tuple: (image as NumPy array, filename, ground truth as NumPy array)
        """
        return self.images[idx], self.filenames[idx], self.gt_images[idx]
    
class ADESet:
    def __init__(self, dataset_dir=None, images=None, gt_images=None, use_gt=False):
        """
        Load and convert ADE20K dataset images to NumPy arrays.
        
        Args:
            dataset_dir (str, optional): Path to ADE20K dataset directory
                             (e.g. "Projects/mmsegmentation/datasets/ade")
                             
                             ADE20K dataset directory structure:
                             ade20k/
                             ├── images/
                             │   ├── validation/
                             │   │   ├── ADE_val_00000001.png
                             │   └── training/
                             └── annotations/
                                 ├── validation/
                                 │   ├── ADE_val_00000001.png
                                 └── training/
                                 
            images (list, optional): List of images as NumPy arrays
        """
        self.use_gt = use_gt
        self.benign_pred = None
        if dataset_dir is not None:
            # Verify ADE20K dataset structure
            images_dir = os.path.join(dataset_dir, "images")
            if not os.path.exists(images_dir):
                raise ValueError(f"Invalid ADE20K dataset structure. {images_dir} does not exist.")
                
            self.dataset_dir = os.path.join(dataset_dir, "images", "validation")
            if not os.path.exists(self.dataset_dir):
                raise ValueError(f"Validation dataset not found at: {self.dataset_dir}")
                
            self.gt_dir = os.path.join(dataset_dir, "annotations", "validation")
            if not os.path.exists(self.gt_dir):
                raise ValueError(f"Ground truth directory not found at: {self.gt_dir}")
                
            self.images, self.filenames, self.gt_images = self._load_images()
        elif images is not None:
            self.images = images
            # Generate filenames as image_0.png, image_1.png, etc.
            self.filenames = [f"image_{i}.png" for i in range(len(images))]
            # For images provided directly, we don't have ground truth
            self.gt_images = gt_images
        else:
            raise ValueError("Either dataset_dir or images must be provided")
    
    def _load_images(self):
        """
        Load all images from ADE20K dataset and convert to list of NumPy arrays.
        Images are converted to BGR format.
        
        Returns:
            tuple: (list of images as NumPy arrays in BGR format, list of corresponding filenames, list of ground truth images)
        """
        images = []
        filenames = []
        gt_images = []
        
        # Load all images in the validation directory
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(self.dataset_dir, filename)
                
                try:
                    # Load image and convert to NumPy array
                    img = Image.open(file_path)
                    img_array = np.array(img)
                    
                    # RGB -> BGR conversion
                    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                        img_array = img_array[:, :, ::-1]
                    
                    images.append(img_array)
                    filenames.append(filename)
                    
                    # Load corresponding ground truth (always .png)
                    gt_filename = os.path.splitext(filename)[0] + ".png"
                    gt_path = os.path.join(self.gt_dir, gt_filename)
                    gt_img = Image.open(gt_path)
                    gt_array = np.array(gt_img)
                    gt_images.append(gt_array)
                        
                except Exception as e:
                    print(f"Error loading image ({filename}): {e}")
        
        if not images:
            print("Warning: No images loaded. Please check dataset path.")
        
        
        return images, filenames, gt_images
    
    
    def __len__(self):
            """
            Return the number of images in the dataset.
            """
            return len(self.images)
        
    def __getitem__(self, idx):
        """
        Return the image, filename and ground truth at the specified index.
        
        Args:
            idx (int): Dataset index
            
        Returns:
            tuple: (image as NumPy array, filename, ground truth as NumPy array)
        """
        return self.images[idx], self.filenames[idx], self.gt_images[idx]

class VOCSet:
    def __init__(self, dataset_dir=None, images=None, gt_images=None, use_gt=False):
        """
        Load and convert VOC2012 dataset images to NumPy arrays.
        
        Args:
            dataset_dir (str, optional): Path to VOC2012 dataset directory
                            (e.g. "Projects/mmsegmentation/datasets/voc")
                            
                            VOC2012 dataset directory structure:
                            VOC2012/
                            ├── JPEGImages/
                            │   ├── 2007_000027.jpg
                            └── SegmentationClass/
                                ├── 2007_000027.png
                                
            images (list, optional): List of images as NumPy arrays
        """
        self.use_gt = use_gt
        self.benign_pred = None
        if dataset_dir is not None:
            # Verify VOC2012 dataset structure
            images_dir = os.path.join(dataset_dir, "JPEGImages")
            if not os.path.exists(images_dir):
                raise ValueError(f"Invalid VOC2012 dataset structure. {images_dir} does not exist.")
                
            self.dataset_dir = os.path.join(dataset_dir, "JPEGImages")
            if not os.path.exists(self.dataset_dir):
                raise ValueError(f"Validation dataset not found at: {self.dataset_dir}")
                
            self.gt_dir = os.path.join(dataset_dir, "SegmentationClass")
            if not os.path.exists(self.gt_dir):
                raise ValueError(f"Ground truth directory not found at: {self.gt_dir}")
                
            self.images, self.filenames, self.gt_images = self._load_images()
        elif images is not None:
            self.images = images
            # Generate filenames as image_0.png, image_1.png, etc.
            self.filenames = [f"image_{i}.png" for i in range(len(images))]
            # For images provided directly, we don't have ground truth
            self.gt_images = gt_images
        else:
            raise ValueError("Either dataset_dir or images must be provided")

    def _load_images(self):
        """
        Load all images from ADE20K dataset and convert to list of NumPy arrays.
        Images are converted to BGR format.
        
        Returns:
            tuple: (list of images as NumPy arrays in BGR format, list of corresponding filenames, list of ground truth images)
        """
        images = []
        filenames = []
        gt_images = []
        
        # Load all images in the validation directory
        for filename in os.listdir(self.dataset_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(self.dataset_dir, filename)
                
                try:
                    # Load image and convert to NumPy array
                    img = Image.open(file_path)
                    img_array = np.array(img)
                    
                    # RGB -> BGR conversion
                    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                        img_array = img_array[:, :, ::-1]
                    
                    images.append(img_array)
                    filenames.append(filename)
                    
                    # Load corresponding ground truth (always .png)
                    gt_filename = os.path.splitext(filename)[0] + ".png"
                    gt_path = os.path.join(self.gt_dir, gt_filename)
                    gt_img = Image.open(gt_path)
                    gt_array = np.array(gt_img)
                    gt_images.append(gt_array)
                        
                except Exception as e:
                    print(f"Error loading image ({filename}): {e}")
        
        if not images:
            print("Warning: No images loaded. Please check dataset path.")
            
        return images, filenames, gt_images
    def __len__(self):
            """
            Return the number of images in the dataset.
            """
            return len(self.images)
        
    def __getitem__(self, idx):
        """
        Return the image, filename and ground truth at the specified index.
        
        Args:
            idx (int): Dataset index
            
        Returns:
            tuple: (image as NumPy array, filename, ground truth as NumPy array)
        """
        return self.images[idx], self.filenames[idx], self.gt_images[idx]


class MedicalNpySet:
    def __init__(self, dataset_dir=None, images=None, gt_images=None, use_gt=True, bbox_shift=20):
        """
        Load medical dataset stored as NumPy arrays.

        Expected directory structure:
            dataset_dir/
            ├── imgs/
            │   ├── xxx.npy
            └── gts/
                ├── xxx.npy

        Args:
            dataset_dir (str, optional): Path to npy dataset root (e.g. data/npy/CT_Abd)
            images (list, optional): List of image arrays
            gt_images (list, optional): List of GT arrays (same order as images)
            use_gt (bool): Whether to load/use GT
            bbox_shift (int): GT bbox   margin 
        """
        self.use_gt = use_gt
        self.bbox_shift = bbox_shift
        self.benign_pred = None
        self.bboxes = []
        self.class_ids = []

        if dataset_dir is not None:
            self.dataset_dir = dataset_dir
            self.img_dir = os.path.join(dataset_dir, "imgs")
            self.gt_dir = os.path.join(dataset_dir, "gts")

            if not os.path.exists(self.img_dir):
                raise ValueError(f"Invalid medical npy dataset structure. {self.img_dir} does not exist.")
            if use_gt and not os.path.exists(self.gt_dir):
                raise ValueError(f"Ground truth directory not found at: {self.gt_dir}")

            self.images, self.filenames, self.gt_images = self._load_images()
        elif images is not None:
            self.images = images
            self.filenames = [f"image_{i}.npy" for i in range(len(images))]
            if gt_images is not None:
                self.gt_images = gt_images
            else:
                self.gt_images = [None] * len(images)
            self._build_bbox_cache()
        else:
            raise ValueError("Either dataset_dir or images must be provided")

    def _get_bboxes_and_labels_from_gt(self, gt, bbox_shift=None):
        """
         GT  label bbox class id .
        bbox : [x_min, y_min, x_max, y_max]
        """
        if gt is None:
            return np.zeros((0, 4), dtype=np.int64), np.zeros((0,), dtype=np.int64)

        if bbox_shift is None:
            bbox_shift = self.bbox_shift

        label_ids = np.unique(gt)
        label_ids = label_ids[label_ids > 0]
        H, W = gt.shape[:2]
        boxes = []
        classes = []

        for lb in label_ids:
            y_indices, x_indices = np.where(gt == lb)
            if len(x_indices) == 0:
                continue
            x_min = max(0, int(np.min(x_indices)) - bbox_shift)
            x_max = min(W, int(np.max(x_indices)) + bbox_shift)
            y_min = max(0, int(np.min(y_indices)) - bbox_shift)
            y_max = min(H, int(np.max(y_indices)) + bbox_shift)
            boxes.append([x_min, y_min, x_max, y_max])
            classes.append(int(lb))

        if not boxes:
            return np.zeros((0, 4), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        return np.array(boxes, dtype=np.int64), np.array(classes, dtype=np.int64)

    def _build_bbox_cache(self):
        """
        self.gt_images   bbox/class  .
        """
        self.bboxes = []
        self.class_ids = []
        for gt in self.gt_images:
            boxes, class_ids = self._get_bboxes_and_labels_from_gt(gt)
            self.bboxes.append(boxes)
            self.class_ids.append(class_ids)

    def _load_images(self):
        """
        Load paired npy samples from imgs/ and gts/.

        Returns:
            tuple: (images, filenames, gt_images)
        """
        images = []
        filenames = []
        gt_images = []

        # gts  ( MedSAM npy  )
        gt_paths = []
        for root, _, files in os.walk(self.gt_dir):
            for name in files:
                if name.endswith(".npy"):
                    gt_paths.append(os.path.join(root, name))
        gt_paths.sort()

        for gt_path in gt_paths:
            name = os.path.basename(gt_path)
            img_path = os.path.join(self.img_dir, name)
            if not os.path.isfile(img_path):
                continue
            try:
                img_arr = np.load(img_path, allow_pickle=True)
                gt_arr = np.load(gt_path, allow_pickle=True)
                images.append(img_arr)
                filenames.append(name)
                gt_images.append(gt_arr)
            except Exception as e:
                print(f"Error loading npy pair ({name}): {e}")

        if not images:
            print("Warning: No medical npy samples loaded. Please check dataset path.")

        self.gt_images = gt_images
        self._build_bbox_cache()
        return images, filenames, gt_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.filenames[idx], self.gt_images[idx]

    def get_item_with_boxes(self, idx):
        """
        bbox/class    .
        Returns:
            tuple: (image, filename, gt, boxes, class_ids)
        """
        return (
            self.images[idx],
            self.filenames[idx],
            self.gt_images[idx],
            self.bboxes[idx],
            self.class_ids[idx],
        )


### test code
if __name__ == "__main__":
    # Test CitySet
    ds = MedicalNpySet("./datasets/CT_Abd", use_gt=True)
    for i in range(3):
        _, name, __, boxes, class_ids = ds.get_item_with_boxes(i)
        print(i, name, _.shape, __.shape, boxes.shape, class_ids.tolist())
    # print("Testing CitySet...")
    # dataset_dir = "./datasets/cityscapes"
    # city_set = CitySet(dataset_dir)
    # print(f"Loaded {len(city_set)} images")
    # image, filename, gt = city_set[0]
    # print(f"Image shape: {image.shape}")
    # print(f"Filename: {filename}")
    # print(f"Ground truth shape: {gt.shape}")
    
    # # Test ADESet
    # print("\nTesting ADESet...")
    # dataset_dir = "./datasets/ade20k"
    # ade_set = ADESet(dataset_dir)
    # print(f"Loaded {len(ade_set)} images")
    # image, filename, gt = ade_set[0]
    # print(f"Image shape: {image.shape}")
    # print(f"Filename: {filename}")
    # print(f"Ground truth shape: {gt.shape}")
    
    # # Test VOCSet
    # print("\nTesting VOCSet...")
    # dataset_dir = "./datasets/VOC2012"
    # voc_set = VOCSet(dataset_dir)
    # print(f"Loaded {len(voc_set)} images")
    # image, filename, gt = voc_set[0]
    # print(f"Image shape: {image.shape}")
    # print(f"Filename: {filename}")
    # print(f"Ground truth shape: {gt.shape}")
    
