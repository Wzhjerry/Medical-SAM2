import os
import cv2
import numpy as np
import torch
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from func_2d.utils import build_transform, random_click
from sklearn.model_selection import KFold
from torchvision.transforms import functional as F
import pandas as pd



os.environ["OPENCV_LOG_LEVEL"] = "0"


class Relabel(Dataset):
    def __init__(self, args, split):
        super(Relabel, self).__init__()
        self.args = args
        self.args.size = 1024
        self.args.sub_data = [
            "DRIVE", 
            "HRF", 
            "GAMMA - task3", 
            "IDRiD"
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
        image = cv2.imread(self.x[idx])[..., ::-1]
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        # name
        name = self.names[idx]
        # label
        target = self.read_labels(self.y[idx], name, self.split)
        if 'vessel' in self.args.exp_name:
            target = target[0]
        elif 'od' in self.args.exp_name:
            target = target[1]
            target[np.where(target > 0)] = 255
        elif 'oc' in self.args.exp_name:
            target = target[1]
            target[np.where(target > 1)] = 255
            target[np.where(target == 1)] = 0
        elif 'ex' in self.args.exp_name:
            target = target[2]
            # target[np.where(target > 0)] = 255
            target = target * 255
        elif 'he' in self.args.exp_name:
            target = target[2]
            target[np.where(target == 2)] = 255
        # target = target[2]
        # target = target * 255

        im = Image.fromarray(np.uint8(image))
        mask = Image.fromarray(np.uint8(target)).convert('L')

        newsize = (1024, 1024)
        mask = mask.resize(newsize)

        point_label, pt = random_click(np.array(mask) / 255, point_label=1)

        # identical transformation for im and gt
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        random.seed(seed)
        im_t = self.im_transform(im)
        torch.manual_seed(seed)
        target_t = self.label_transform(mask)
        torch.manual_seed(seed)
        random.seed(seed)
        image_meta_dict = {'filename_or_obj': name}

        return {
            'image': im_t,              # Transformed image (tensor)
            'mask': target_t,          # Transformed multi-class mask (tensor)
            'p_label': point_label,  # Tensor of point labels (num_classes,)
            'pt': pt,            # Tensor of points (num_classes, 2)
            'image_meta_dict': image_meta_dict,
        }

    def read_labels(self, root_dirs, name, split):
        # Read labels for vessel seg

        target_vessel = Image.open(root_dirs[0])
        target_vessel = np.array(target_vessel).astype(np.uint8)
        if len(target_vessel.shape) == 3:
            target_vessel = target_vessel[..., 0]
        target_vessel = cv2.resize(target_vessel, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)

        # Convert label from numpy to Image
        # target_vessel = Image.fromarray(np.uint8(target_vessel)).convert('1')

        target_odoc = Image.open(root_dirs[1])
        target_odoc = np.array(target_odoc).astype(np.uint8)
        if len(target_odoc.shape) == 3:
            target_odoc = target_odoc[..., 0]

        target_return = np.zeros_like(target_odoc, dtype=np.uint8)
        # target_return[np.where(0 < target_odoc < 255)] = 1
        # target_return[np.where(target_odoc == 255)] = 2
        target_odoc[(target_odoc > 0) & (target_odoc < 255)] = 1
        target_odoc[target_odoc == 255] = 2
        target_odoc = cv2.resize(target_odoc, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)
        
        # target_odoc = Image.fromarray(np.uint8(target_odoc))
        
        target_ex = Image.open(root_dirs[2])
        target_ex = np.array(target_ex).astype(np.uint8)
        if len(target_ex.shape) == 3:
            target_ex = target_ex[..., 0]
        target_ex = cv2.resize(target_ex, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)

        target_he = Image.open(root_dirs[3])
        target_he = np.array(target_he).astype(np.uint8)
        if len(target_he.shape) == 3:
            target_he = target_he[..., 0]
        target_he = cv2.resize(target_he, (self.args.size, self.args.size), interpolation=cv2.INTER_NEAREST)

        target_lesion = np.zeros_like(target_ex, dtype=np.uint8)
        target_lesion[np.where(target_ex > 0)] = 1
        # target_lesion[np.where(target_he > 0)] = 2
        
        # target_lesion = Image.fromarray(np.uint8(target_lesion))

        return [target_vessel, target_odoc, target_lesion]

    def load_name(self, args, split):
        inputs, targets, names = [], [], []

        vessel_inputs, vessel_targets, vessel_names = self.load_vessel(args, split)
        ODOC_inputs, ODOC_targets, ODOC_names = self.load_ODOC(args, split)
        lesion_inputs, lesion_targets, lesion_names = self.load_lesion(args, split)

        for i in range(len(vessel_inputs)):
            inputs.append(vessel_inputs[i])
            targets.append([vessel_targets[0][i], vessel_targets[1][i], vessel_targets[2][i], vessel_targets[3][i]])
            names.append(vessel_names[i])
        
        for i in range(len(ODOC_inputs)):
            inputs.append(ODOC_inputs[i])
            targets.append([ODOC_targets[0][i], ODOC_targets[1][i], ODOC_targets[2][i], ODOC_targets[3][i]])
            names.append(ODOC_names[i])

        for i in range(len(lesion_inputs)):
            inputs.append(lesion_inputs[i])
            targets.append([lesion_targets[0][i], lesion_targets[1][i], lesion_targets[2][i], lesion_targets[3][i]])
            names.append(lesion_names[i])
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        names = np.array(names)

        return inputs, targets, names

    def load_vessel(self, args, split):
        inputs, targets_vessel, targets_odoc, targets_ex, targets_he, names = [], [], [], [], [], []

        # Define root directories
        root_dir = '/data/wangzh/datasets/fundus_miccai/vessel_seg/'

        for dataset in args.sub_data:
            csv_path = os.path.join(root_dir, f'{dataset}/{split}_relabel.csv')

            if not os.path.exists(csv_path):
                continue
            data = pd.read_csv(csv_path)
            images = data['image_path'].values
            labels_vessel = data['label_vessel'].values
            labels_odoc = data['label_odoc'].values
            labels_ex = data['label_ex'].values
            labels_he = data['label_he'].values
            image_names = data['image_name'].values
            image_names = [dataset + '_' + split + '_' + str(name) for name in image_names]

            inputs.extend(images)
            targets_vessel.extend(labels_vessel)
            targets_odoc.extend(labels_odoc)
            targets_ex.extend(labels_ex)
            targets_he.extend(labels_he)
            names.extend(image_names)

        inputs = [str.replace('/datasets/vessel_seg/', root_dir) for str in inputs]
        targets_vessel = [str.replace('/datasets/vessel_seg/', root_dir) for str in targets_vessel]
        targets_odoc = [str.replace('/datasets/vessel_seg/', root_dir) for str in targets_odoc]
        targets_ex = [str.replace('/datasets/vessel_seg/', root_dir) for str in targets_ex]
        targets_he = [str.replace('/datasets/vessel_seg/', root_dir) for str in targets_he]

        inputs = np.array(inputs)
        targets_vessel = np.array(targets_vessel)
        targets_odoc = np.array(targets_odoc)
        targets_ex = np.array(targets_ex)
        targets_he = np.array(targets_he)
        names = np.array(names)

        print("=> Using {} images for vessel {}".format(len(inputs), split))
        return inputs, [targets_vessel, targets_odoc, targets_ex, targets_he], names

    def load_ODOC(self, args, split):
        inputs, targets_vessel, targets_odoc, targets_ex, targets_he, names = [], [], [], [], [], []

        # Define root directories
        root_dir = '/data/wangzh/datasets/fundus_miccai/ODOC_seg/'

        for dataset in args.sub_data:
            csv_path = os.path.join(root_dir, f'{dataset}/{split}_relabel.csv')

            if not os.path.exists(csv_path):
                continue
            data = pd.read_csv(csv_path)
            images = data['image_path'].values
            labels_vessel = data['label_vessel'].values
            labels_odoc = data['label_odoc'].values
            labels_ex = data['label_ex'].values
            labels_he = data['label_he'].values
            image_names = data['image_name'].values
            image_names = [dataset + '_' + split + '_' + str(name) for name in image_names]

            inputs.extend(images)
            targets_vessel.extend(labels_vessel)
            targets_odoc.extend(labels_odoc)
            targets_ex.extend(labels_ex)
            targets_he.extend(labels_he)
            names.extend(image_names)

        inputs = [str.replace('/datasets/ODOC_seg/', root_dir) for str in inputs]
        targets_vessel = [str.replace('/datasets/ODOC_seg/', root_dir) for str in targets_vessel]
        targets_odoc = [str.replace('/datasets/ODOC_seg/', root_dir) for str in targets_odoc]
        targets_ex = [str.replace('/datasets/ODOC_seg/', root_dir) for str in targets_ex]
        targets_he = [str.replace('/datasets/ODOC_seg/', root_dir) for str in targets_he]

        inputs = np.array(inputs)
        targets_vessel = np.array(targets_vessel)
        targets_odoc = np.array(targets_odoc)
        targets_ex = np.array(targets_ex)
        targets_he = np.array(targets_he)
        names = np.array(names)

        print("=> Using {} images for ODOC {}".format(len(inputs), split))
        return inputs, [targets_vessel, targets_odoc, targets_ex, targets_he], names

    def load_lesion(self, args, split):
        inputs, targets_vessel, targets_odoc, targets_ex, targets_he, names = [], [], [], [], [], []

        # Define root directories
        root_dir = '/data/wangzh/datasets/fundus_miccai/lesion_seg/'

        for dataset in args.sub_data:
            csv_path = os.path.join(root_dir, f'{dataset}/{split}_relabel.csv')

            if not os.path.exists(csv_path):
                continue
            data = pd.read_csv(csv_path)
            images = data['image_path'].values
            labels_vessel = data['label_vessel'].values
            labels_odoc = data['label_odoc'].values
            labels_ex = data['label_ex'].values
            labels_he = data['label_he'].values
            image_names = data['image_name'].values
            image_names = [dataset + '_' + split + '_' + str(name) for name in image_names]

            inputs.extend(images)
            targets_vessel.extend(labels_vessel)
            targets_odoc.extend(labels_odoc)
            targets_ex.extend(labels_ex)
            targets_he.extend(labels_he)
            names.extend(image_names)

        inputs = [str.replace('/datasets/lesion_seg/', root_dir) for str in inputs]
        targets_vessel = [str.replace('/datasets/lesion_seg/', root_dir) for str in targets_vessel]
        targets_odoc = [str.replace('/datasets/lesion_seg/', root_dir) for str in targets_odoc]
        targets_ex = [str.replace('/datasets/lesion_seg/', root_dir) for str in targets_ex]
        targets_he = [str.replace('/datasets/lesion_seg/', root_dir) for str in targets_he]

        inputs = np.array(inputs)
        targets_vessel = np.array(targets_vessel)
        targets_odoc = np.array(targets_odoc)
        targets_ex = np.array(targets_ex)
        targets_he = np.array(targets_he)
        names = np.array(names)

        print("=> Using {} images for Lesion {}".format(len(inputs), split))
        return inputs, [targets_vessel, targets_odoc, targets_ex, targets_he], names


def load_dataset(args, train=False):
    if train:
        train_dataset = Relabel(args, 'train')
        val_dataset = Relabel(args, 'val')
        return train_dataset, val_dataset
    else:
        test_dataset = Relabel(args, 'test')
        return test_dataset
