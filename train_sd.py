import os
import cv2
import sys
import tqdm
import shutil
import random
import logging
import argparse
import numpy as np
import setproctitle
import matplotlib.pyplot as plt
from skimage.color import label2rgb
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from models.sdnet import SDNet
from dataset.abus_dataset import ABUS_2D, ElasticTransform, ToTensor,  Normalize
from utils.utils import save_checkpoint, confusion
from utils.loss import DiceLoss, KL_divergence


def get_args():
    print('initing args------')
    parser = argparse.ArgumentParser()

    # general config
    parser.add_argument('--gpu', default='2', type=str)
    parser.add_argument('--ngpu', default=1, type=str)
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--start_epoch', default=1, type=int)

    # dataset config
    parser.add_argument('--root_path', default='/data/xuyangcao/code/data/abus_2d/', type=str)
    parser.add_argument('--sample_k', '-k', default=8856, type=int, choices=(100, 300, 885, 1770, 4428, 8856)) # 8856 if supervised
    parser.add_argument('--batch_size', type=int, default=1)

    # optimizer config
    parser.add_argument('--lr', default=1e-4, type=float) 
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # network config
    parser.add_argument('--arch', default='sdnet', type=str, choices=('dense161', 'dense121', 'dense201', 'unet', 'resunet', 'sdnet'))
    parser.add_argument('--drop_rate', default=0.3, type=float)

    # decomposition config 
    parser.add_argument('--anatomy_factors', type=int, default=4)
    parser.add_argument('--modality_factors', type=int, default=8)

    # regularizers weights
    parser.add_argument('--kl_w', type=float, default=0.01)
    parser.add_argument('--regress_w', type=float, default=1.0)
    parser.add_argument('--focal_w', type=float, default=0.0)
    parser.add_argument('--dice_w', type=float, default=10.0)
    parser.add_argument('--reco_w', type=float, default=1.0)

    # save config
    parser.add_argument('--log_dir', default='./log/sdnet')
    parser.add_argument('--save', default='./work/sdnet/test')

    args = parser.parse_args()
    return args


def main():
    #############
    # init args #
    #############
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    # creat save path
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # logger
    logging.basicConfig(filename=args.save+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('--- init parameters ---')

    # writer
    idx = args.save.rfind('/')
    log_dir = args.log_dir + args.save[idx:]
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # set title of the current process
    setproctitle.setproctitle(args.save)

    # random
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #####################
    # building  network #
    #####################
    logging.info('--- building network ---')
    
    model_params = {
            'width': 256,
            'height': 256,
            'ndf': 64,
            'norm': 'batchnorm',
            'upsample': 'nearest',
            'num_classes': 2,
            'anatomy_out_channels': args.anatomy_factors,
            'z_length': args.modality_factors,
            'num_mask_channels': 4,
            }
    if args.arch == 'sdnet':
        model = SDNet(**model_params)
    else:
        raise(NotImplementedError('model {} not implement'.format(args.arch))) 
    n_params = sum([p.data.nelement() for p in model.parameters()])
    logging.info('--- total parameters = {} ---'.format(n_params))

    model = model.cuda()

    ################
    # prepare data #
    ################
    logging.info('--- loading dataset ---')

    train_transform = transforms.Compose([
        ElasticTransform('train'), 
        ToTensor(mode='train'), 
        Normalize(0.5, 0.5, mode='train')
        ])
    val_transform = transforms.Compose([
        ElasticTransform(mode='val'),
        ToTensor(mode='val'), 
        Normalize(0.5, 0.5, mode='val')
        ])
    train_set = ABUS_2D(base_dir=args.root_path,
                        mode='train', 
                        data_num_labeled=args.sample_k,
                        use_unlabeled_data=False,
                        transform=train_transform
                        )
    val_set = ABUS_2D(base_dir=args.root_path,
                       mode='val', 
                       data_num_labeled=None, 
                       use_unlabeled_data=False, 
                       transform=val_transform
                       )
    kwargs = {'num_workers': 0, 'pin_memory': True} 
    batch_size = args.ngpu * args.batch_size
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, **kwargs)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, worker_init_fn=worker_init_fn, **kwargs)

    #####################
    # optimizer & loss  #
    #####################
    logging.info('--- configing optimizer & losses ---')
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    loss_fn = {}
    loss_fn['dice_loss'] = DiceLoss()
    loss_fn['l1_loss'] = nn.L1Loss()
    loss_fn['kl_loss'] = KL_divergence


    #####################
    #   strat training  #
    #####################
    logging.info('--- start training ---')

    best_pre = 0.
    nTrain = len(train_set)
    for epoch in range(args.start_epoch, args.n_epochs + 1):
        # update learning rate
        if epoch % 30 == 0:
            if epoch % 60 == 0:
                lr *= 0.2
            else:
                lr *= 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train(args, epoch, model, train_loader, optimizer, loss_fn, writer)
        if epoch == 1 or epoch % 5 == 0:
            dice = val(args, epoch, model, val_loader, optimizer, loss_fn, writer)
            is_best = False
            if dice > best_pre:
                is_best = True
                best_pre = dice

            if is_best or epoch % 10 == 0:
                save_checkpoint({'epoch': epoch,
                                 'state_dict': model.state_dict(),
                                 'best_pre': best_pre},
                                  is_best, 
                                  args.save, 
                                  args.arch)
    writer.close()



def train(args, epoch, model, train_loader, optimizer, loss_fn, writer):
    model.train()
    nProcessed = 0
    batch_size = args.ngpu * args.batch_size
    nTrain = len(train_loader.dataset)
    reco_loss_list = []
    kl_loss_list = []
    dice_loss_list = []
    regression_loss_list = []
    loss_list = []

    for batch_idx, sample in enumerate(train_loader):
        # read data
        image, target = sample['image'], sample['target']
        image, target = Variable(image.cuda()), Variable(target.cuda(), requires_grad=False)

        # forward
        reco, z_out, mu_tilde, a_mu_tilde, a_out, seg_pred, mu, logvar, a_mu, a_logvar = model(image, target, 'training')

        # backward
        reco_loss = loss_fn['l1_loss'](reco, image)
        kl_loss = loss_fn['kl_loss'](logvar, mu)
        dice_loss = loss_fn['dice_loss'](seg_pred, target)
        regression_loss = loss_fn['l1_loss'](mu_tilde, z_out)
        loss = args.reco_w * reco_loss\
                + args.kl_w * kl_loss\
                + args.dice_w * dice_loss\
                + args.regress_w * regression_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reco_loss_list.append(reco_loss.item())
        kl_loss_list.append(kl_loss.item())
        dice_loss_list.append(dice_loss.item())
        regression_loss_list.append(regression_loss.item())
        loss_list.append(loss.item())

        # visualization
        nProcessed += len(image)
        partialEpoch = epoch + batch_idx / len(train_loader)
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(train_loader),
            loss.item()))

        with torch.no_grad():
            padding = 10
            nrow = 5
            if batch_idx % 10 == 0:
                # 1. show gt and prediction
                image = (image * 0.5 + 0.5)
                img = make_grid(image, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 255
                img = img.astype(np.uint8)
                gt = make_grid(target, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                gt_img = label2rgb(gt, img, bg_label=0)
                pre = torch.max(seg_pred, dim=1, keepdim=True)[1]
                pre = pre.float()
                pre = make_grid(pre, nrow=nrow, padding=padding).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                pre_img = label2rgb(pre, img, bg_label=0)
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(gt_img)
                ax.set_title('train ground truth')
                ax = fig.add_subplot(212)
                ax.imshow(pre_img)
                ax.set_title('train prediction')
                fig.tight_layout() 
                writer.add_figure('train_result', fig, epoch)
                fig.clear()

    writer.add_scalar('reco_loss', float(np.mean(reco_loss_list)), epoch)
    writer.add_scalar('kl_loss', float(np.mean(kl_loss_list)), epoch)
    writer.add_scalar('dice_loss', float(np.mean(dice_loss_list)), epoch)
    writer.add_scalar('regression_loss', float(np.mean(regression_loss_list)), epoch)
    writer.add_scalar('total_loss',float(np.mean(loss_list)), epoch)


def val(args, epoch, model, val_loader, optimizer, loss_fn, writer):
    model.eval()
    dice_list = []

    with torch.no_grad():
        for batch_idx, sample in tqdm.tqdm(enumerate(val_loader)):
            image, target = sample['image'], sample['target']
            image, target = image.cuda(), target.cuda()

            # forward
            _, _, _, _, _, seg_pred, _, _, _, _ = model(image, target, 'val')                
            dice = DiceLoss.dice_coeficient(seg_pred, target)
            dice_list.append(dice.item())

            # visualization 
            if batch_idx % 10 == 0:
                padding = 10
                nrow = 5
                image = (image * 0.5 + 0.5)
                img = make_grid(image, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * 255
                img = img.astype(np.uint8)
                gt = make_grid(target, nrow=nrow, padding=padding).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0]
                gt_img = label2rgb(gt, img, bg_label=0)
                pre = torch.max(out, dim=1, keepdim=True)[1]
                pre = pre.float()
                pre = make_grid(pre, nrow=nrow, padding=padding).cpu().numpy().transpose(1, 2, 0)[:, :, 0]
                pre_img = label2rgb(pre, img, bg_label=0)
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.imshow(gt_img)
                ax.set_title('train ground truth')
                ax = fig.add_subplot(212)
                ax.imshow(pre_img)
                ax.set_title('train prediction')
                fig.tight_layout() 
                writer.add_figure('train_result', fig, epoch)
                fig.clear()

        writer.add_scalar('val_dice', float(np.mean(dice_list)), epoch)
        return np.mean(dice_list)


if __name__ == '__main__':
    main()
