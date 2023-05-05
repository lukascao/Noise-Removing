from visualdl import LogWriter
import os
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from data.dataloader import ErasingData, devdata
from loss.Loss import LossWithGAN_STE
from models.sa_aidr import STRAIDR
import utils
import random
import numpy as np

log = LogWriter('log')
# 多卡训练
# python3 -m paddle.distributed.launch train.py

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=8, help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='', help='path for saving models')
parser.add_argument('--logPath', type=str, default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512, help='image loading size')
parser.add_argument('--dataRoot', type=str, default='')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=5000, help='epochs')
parser.add_argument('--net', type=str, default='str')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--lr_decay_iters', type=int, default=300000, help='learning rate decay per N iters')
parser.add_argument('--mode', type=str, default='l1')
parser.add_argument('--seed', type=int, default=2022)
args = parser.parse_args()

log_file = os.path.join('./log', args.net + '_log.txt')
logging = utils.setup_logger(output=log_file, name=args.net)
logging.info(args)

# set gpu
if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu:0')
    # paddle.set_device('gpu')
else:
    paddle.set_device('cpu')
# paddle.distributed.init_parallel_env()
# set random seed
logging.info('========> Random Seed: {}'.format(args.seed))
random.seed(args.seed)
np.random.seed(args.seed)
paddle.seed(args.seed)
paddle.framework.random._manual_program_seed(args.seed)


batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
save_dir = os.path.join(args.modelsSavePath, args.net)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dataRoot = args.dataRoot

Erase_data = ErasingData(dataRoot, loadSize, mode='train')
Erase_data = DataLoader(Erase_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False)

Erase_val_data = ErasingData(dataRoot, loadSize, mode='val')
Erase_val_data = DataLoader(Erase_val_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
print('==============', len(Erase_data) * batchSize, len(Erase_val_data))
print('==============>net use: ', args.net)
if 'idr' in args.net:
    netG = STRAIDR(num_c=96)

if args.pretrained != '':
    print('loaded {}'.format(args.pretrained))
    weights = paddle.load(args.pretrained)
    netG.load_dict(weights)

# netG = paddle.DataParallel(netG)
count = 1
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=args.lr_decay_iters, gamma=args.gamma, verbose=False)
G_optimizer = paddle.optimizer.Adam(scheduler, parameters=netG.parameters(), weight_decay=0.0)#betas=(0.5, 0.9))

criterion = LossWithGAN_STE(lr=0.00001, betasInit=(0.0, 0.9), Lamda=10.0, mode=args.mode)
print('OK!')
num_epochs = args.num_epochs
mse = nn.MSELoss()
best_psnr = 0
iters = 0
for epoch in range(1, num_epochs + 1):
    netG.train()

    for k, (imgs, gt, masks, path) in enumerate(Erase_data):
        iters += 1
        # print(imgs.shape, gt.shape, masks.shape)

        x_o1, x_o2, x_o3, fake_images, mm = netG(imgs)
        # print(mm.shape)
        G_loss = criterion(imgs, masks, x_o1, x_o2, x_o3, fake_images, mm, gt, count, epoch)
        G_loss = G_loss.sum()
        G_optimizer.clear_grad()
        G_loss.backward()
        G_optimizer.step()
        scheduler.step()
        if iters % 10 == 0:
            logging.info('[{}/{}] Generator Loss of epoch{} is {:.5f}, {},  Lr:{}'.format(iters, len(Erase_data) * num_epochs, epoch, G_loss.item(), args.net, G_optimizer.get_lr()))
            log.add_scalar(tag="train_loss", step=iters, value=G_loss.item())
        count += 1
    
        if iters % 200 == 0:
            netG.eval()
            val_psnr = 0
            for index, (imgs, gt, masks, path) in enumerate(Erase_val_data):
                print(index, imgs.shape, gt.shape, path)
                _,_,h,w = imgs.shape
                rh, rw = h, w
                step = 512
                pad_h = step - h if h < step else 0
                pad_w = step - w if w < step else 0
                m = nn.Pad2D((0, pad_w,0, pad_h))
                imgs = m(imgs)
                _, _, h, w = imgs.shape
                res = paddle.zeros_like(imgs)
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        if h - i < step:
                            i = h - step
                        if w - j < step:
                            j = w - step
                        clip = imgs[:, :, i:i+step, j:j+step]
                        clip = clip.cuda()
                        with paddle.no_grad():
                            _, _, _, g_images_clip,mm = netG(clip)
                        g_images_clip = g_images_clip.cpu()
                        mm = mm.cpu()
                        clip = clip.cpu()
                        mm = paddle.where(F.sigmoid(mm)>0.5, paddle.zeros_like(mm), paddle.ones_like(mm))
                        g_image_clip_with_mask = clip * (mm) + g_images_clip * (1- mm)
                        res[:, :, i:i+step, j:j+step] = g_image_clip_with_mask
                res = res[:, :, :rh, :rw]
                output = utils.pd_tensor2img(res)
                target = utils.pd_tensor2img(gt)
                del res
                del gt
                # psnr = utils.compute_psnr(target, output)
                psnr = utils.calculate_psnr(target, output)
                del target
                del output
                val_psnr += psnr
                logging.info('index:{} psnr: {}'.format(index, psnr))
            ave_psnr = val_psnr/(index+1)
            log.add_scalar(tag="valid_psnr", step=iters, value=ave_psnr)
            paddle.save(netG.state_dict(), save_dir + '/{}_{:.4f}.pdparams'.format(epoch, ave_psnr))
            if ave_psnr > best_psnr:
                best_psnr = ave_psnr
                paddle.save(netG.state_dict(), save_dir + '/model_best.pdparams')
            logging.info('epoch: {}, ave_psnr: {}, best_psnr: {}'.format(epoch, ave_psnr, best_psnr))
