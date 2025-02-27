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

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)

    # optimisation
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

    '''load pretrained model'''

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)


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

    elif args.dataset == 'relabel':
        from func_2d.relabel import Relabel
        relabel_train_dataset = Relabel(args, split = 'train')
        relabel_test_dataset = Relabel(args, split = 'test')

        nice_train_loader = DataLoader(multitask_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(multitask_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'vessel':
        from func_2d.vessel import Vessel
        vessel_train_dataset = Vessel(args, split = 'train')
        vessel_test_dataset = Vessel(args, split = 'val')

        nice_train_loader = DataLoader(vessel_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(vessel_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'od':
        from func_2d.od import OD
        od_train_dataset = OD(args, split = 'train')
        od_test_dataset = OD(args, split = 'val')

        nice_train_loader = DataLoader(od_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(od_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'oc':
        from func_2d.oc import OC
        oc_train_dataset = OC(args, split = 'train')
        oc_test_dataset = OC(args, split = 'val')

        nice_train_loader = DataLoader(oc_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(oc_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'ex':
        from func_2d.ex import EX
        ex_train_dataset = EX(args, split = 'train')
        ex_test_dataset = EX(args, split = 'val')

        nice_train_loader = DataLoader(ex_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(ex_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    elif args.dataset == 'he':
        from func_2d.he import HE
        he_train_dataset = HE(args, split = 'train')
        he_test_dataset = HE(args, split = 'val')

        nice_train_loader = DataLoader(he_train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(he_test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
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


    for epoch in range(settings.EPOCH):

        if epoch == 0:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

        # training
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        # validation
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:

            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if edice > best_dice:
                best_dice = edice
                torch.save({'model': net.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))


    writer.close()


if __name__ == '__main__':
    main()
