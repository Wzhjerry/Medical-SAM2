# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Jiayuan Zhu
"""

import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader

import cfg
import func_2d.function as function
from conf import settings
#from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *


def main():
    # use bfloat16 for the entire work
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    '''load pretrained model'''

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # args.sam_ckpt = os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth')
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    # optimisation
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    
    # example of REFUGE dataset
    if args.dataset == 'REFUGE':
        '''REFUGE data'''
        refuge_train_dataset = REFUGE(args, args.data_path, transform = transform_train, mode = 'Training')
        refuge_test_dataset = REFUGE(args, args.data_path, transform = transform_test, mode = 'Test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
        '''end'''
    elif args.dataset == 'multitask':
        from func_2d.mutitask import Multitask
        multitask_train_dataset = Multitask(args, split = 'train')
        multitask_test_dataset = Multitask(args, split = 'val')

        nice_train_loader = DataLoader(multitask_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(multitask_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'relabel':
        from func_2d.relabel import Relabel
        relabel_train_dataset = Relabel(args, split = 'train')
        relabel_test_dataset = Relabel(args, split = 'test')

        nice_train_loader = DataLoader(relabel_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(relabel_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)


    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    # checkpoint_path = os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth')
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0

    # validation
    net.eval()

    tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, 0, net, writer)
    logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {0}.')

    # if edice > best_dice:
    #     best_dice = edice
    #     torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))


    writer.close()


if __name__ == '__main__':
    main()
