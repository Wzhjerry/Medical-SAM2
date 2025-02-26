import os
import cv2
import numpy as np
import torch
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
# from datasets.utils import analyze_name
from utils import random_box, random_click, build_transform
from sklearn.model_selection import KFold
from torchvision.transforms import functional as F
import pandas as pd



os.environ["OPENCV_LOG_LEVEL"] = "0"


class ODOC(Dataset):
    def __init__(self, args, split):
        super(ODOC, self).__init__()
        self.args = args
        self.args.size = 1024
        self.args.sub_data = [
            # "DRIVE", 
            # "FIVES", 
            # "HRF", 
            # "STARE", 
            "G1020", 
            "GAMMA - task3", 
            "ORIGA", 
            "Papila", 
            "REFUGE", 
            # "DDR - lesion_seg", 
            # "FGADR-Seg-set", 
            # "IDRiD"
        ]

        self.x, self.y, self.names = self.load_name(args, split)
        assert len(self.x) == len(self.y) == len(self.names)
        self.dataset_size = len(self.x)
        self.train = True if split == "train" else False
        self.im_transform, self.label_transform = build_transform(args, self.train)
        self.split = split

    def __len__(self):
        # if self.train:
        #     return self.dataset_size * 2
        # else:
        return self.dataset_size

    def _get_index(self, idx):
        if self.train:
            return idx % self.dataset_size
        else:
            return idx

    def __getitem__(self, idx):
        # BGR -> RGB -> PIL
        point_label = 1
        # image = cv2.imread(self.x[idx])[..., ::-1]
        # image, ymin, ymax, xmin, xmax = remove_black_edge(image)
        # image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # # Name
        # name = self.names[idx]
        # # Label
        # label = self.read_labels(self.y[idx], name, ymin, ymax, xmin, xmax, self.split)
        
        # im = Image.fromarray(np.uint8(image)).convert('RGB')

        image = Image.open(self.x[idx]).convert('RGB')
        mask = Image.open(self.y[idx]).convert('L')

        newsize = (1024, 1024)
        mask = mask.resize(newsize)

        point_label, pt = random_click(np.array(mask) / 255, point_label)

        # Identical transformations for image and ground truth
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)
        im_t = self.im_transform(image)
        torch.manual_seed(seed)
        random.seed(seed)
        target_t = self.label_transform(mask).int()
        torch.manual_seed(seed)
        random.seed(seed)

        image_meta_dict = {'filename_or_obj': self.names[idx]}

        return {
            'image': im_t,              # Transformed image (tensor)
            'mask': target_t,          # Transformed multi-class mask (tensor)
            'p_label': point_label,  # Tensor of point labels (num_classes,)
            'pt': pt,            # Tensor of points (num_classes, 2)
            'image_meta_dict': image_meta_dict,
        }

    def read_labels(self, root_dirs, name, ymin, ymax, xmin, xmax, split):

        label = Image.open(root_dirs)
        label = np.array(label).astype(np.uint8)
        if len(label.shape) == 3:
            label = label[..., 0]
        label = label[ymin:ymax, xmin:xmax]
        # label[(label > 0) & (label < 255)] = 1
        # label[label == 255] = 2
        label[label > 0] = 1
        label = cv2.resize(label, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        label = Image.fromarray(np.uint8(label)).convert("1")

        return label


    def read_images(self, root_dir):
        image_paths = []
        patterns = ['*.png', '*.jpg', '*.jpeg']
        for pattern in patterns:
            full_pattern = os.path.join(root_dir, '**', pattern)
            found_paths = glob.glob(full_pattern, recursive=True)
            image_paths.extend(found_paths)
        return image_paths

    def load_name(self, args, split):

        ODOC_inputs, ODOC_targets, ODOC_names = self.load_ODOC(args, split)
        
        inputs = np.array(ODOC_inputs)
        targets = np.array(ODOC_targets)
        names = np.array(ODOC_names)

        return inputs, targets, names


    def load_ODOC(self, args, split):
        inputs, targets, names = [], [], []

        # Define root directories
        root_dir = '/data/wangzh/datasets/fundus_miccai/ODOC_seg/'

        for dataset in args.sub_data:
            csv_path = os.path.join(root_dir, f'{dataset}/{split}.csv')
            print(csv_path)
            if not os.path.exists(csv_path):
                continue
            data = pd.read_csv(csv_path)
            images = data['image_path'].values
            labels = data['label_path'].values
            image_names = data['image_name'].values
            image_names = [dataset + '_' + split + '_' + str(name) for name in image_names]

            inputs.extend(images)
            targets.extend(labels)
            names.extend(image_names)

        inputs = [str.replace('/datasets/ODOC_seg/', root_dir) for str in inputs]
        targets = [str.replace('/datasets/ODOC_seg/', root_dir) for str in targets]

        inputs = np.array(inputs)
        targets = np.array(targets)
        names = np.array(names)

        print("=> Using {} images for ODOC {}".format(len(inputs), split))
        return inputs, targets, names



