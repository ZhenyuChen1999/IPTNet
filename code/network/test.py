# !pip install pytorch_metric_learning
import argparse
import sys
import random
import numpy as np
import cv2
import time
import logging
import imageio
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
from torch.nn import init
import time
from tqdm import tqdm
import helpers
from create_model import Generator
from pympler import tracker,summary,muppy
memory_tracker = tracker.SummaryTracker()



parser = argparse.ArgumentParser()
parser.add_argument("--mask_dir", default='/data/orbitchen/volumeRender/subsurface/real_scene_standard/mask', help="dir of mask")
parser.add_argument("--val_dir", default='/data/orbitchen/volumeRender/subsurface/real_scene_standard/photographs')
parser.add_argument("--outfile", default='real0222')
parser.add_argument("--model_dir", default='./model')
parser.add_argument("--batch_size", type=int, default=2, help="number of images in batch")
parser.add_argument("--val_batch_size", type=int, default=1, help="number of images in validation batch")
parser.add_argument("--mode", default='test', choices=["test", "train"])
parser.add_argument("--seed", type=int, default=20131130)
parser.add_argument("--nodecay", dest="decay", action="store_false", help="don't use lr decay")
parser.set_defaults(decay=True)

parser.add_argument("--lr", type=float, default=0.00005, help="initial learning rate for adam")
parser.add_argument("--init_epoch", type=int, default=0, help="number of already training epoch")
parser.add_argument("--max_epochs", type=int, default=400, help="number of training epochs")
parser.add_argument("--num_workers", type=int, default=10, help="number of reading images workers")
parser.add_argument("--verbose_step", type=int, default=1, help="number of tqdm refreshing step")
parser.add_argument("--crop_size", type=int, default=1024, help="crop images to this size  ")
parser.add_argument("--down_size", type=int, default=256, help="down images to this size  ")
parser.add_argument("--patch_size", type=int, default=256, help="patch images to this size  ")
parser.add_argument('--no_cuda', action='store_true', default=False)

a = parser.parse_args()
mean=(0.5, 0.5, 0.5)
std=(0.5, 0.5, 0.5)


class RenderDataset(Dataset):
    def __init__(self, inputDir_list, mode='train'):
        self.mode = mode
        self.imgs = []
        self.mask_dir = a.mask_dir

        for inputDir in inputDir_list:
            length = len(os.listdir(inputDir)) - 1
            for i in range(length):
                self.imgs.append(os.path.join(inputDir, str(i)))
        if mode == "test":
            print('test dataset nums:{}'.format(len(self.imgs)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        num = self.imgs[index]
        images = []
        dir = os.path.join(a.val_dir, str(num))
        print(dir)
        exrlist_temp = sorted(os.listdir(dir))

        exrlist = []
        for i in exrlist_temp:
            if '_' in i and '.exr' in i:
                exrlist.append(i)

        # exrlist.remove("params.txt")
        # exrlist.remove("bias.exr")
        # exrlist.remove("albedo.exr")
        exrlist_temp = []
        for i in exrlist:
            if i.count('_') == 1:
                exrlist_temp.append(i)
        exrlist = exrlist_temp
        getnum = 10  #input pics num
        if a.mode=='train':
            selected_list = random.sample(exrlist, getnum)
            # selected_list = exrlist[:getnum]
        else:
            selected_list = exrlist[:getnum]

        for i in selected_list:
            image = imageio.imread(os.path.join(dir, i), 'exr')
            image = np.clip(image,0,1)
            image = image.astype(np.float32)
            images.append(image)
        num = int(os.path.split(dir)[-1])
        s = os.path.join(a.mask_dir, f'mask_{num}.png')
        mask = imageio.imread(s)
        mask = mask.astype(np.float32)
        mask = np.stack((mask, mask, mask), -1)
        images.append(mask)

        image = imageio.imread(os.path.join(dir, 'albedo.exr'), 'exr')
        image = np.clip(image, 0, 1)
        image = image.astype(np.float32)
        images.append(image)
        image = imageio.imread(os.path.join(dir, 'bias.exr'), 'exr')
        image = np.clip(image, 0, 1)
        image = image.astype(np.float32)
        images.append(image)

        images = np.stack(images, axis=0)

        mean = np.array([25, 25, 25, 0.3, 0.3, 0.3], dtype=np.float32)
        std = np.array([275, 275, 275, 0.65, 0.65, 0.65], dtype=np.float32)
        images = torch.from_numpy(images)
        return images,self.imgs[index]


def prepare_dataloader():
    valid_ds = RenderDataset(inputDir_list=[a.val_dir], mode='valid')

    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=a.val_batch_size,
        num_workers=a.num_workers,
        shuffle=False,
    )
    return val_loader


def l1(outputs, targets):
    return torch.mean(torch.abs(outputs-targets))


def test_epoch(epoch, model, val_loader):
    model.eval()

    sigma_sumloss = 0
    rgb_sumloss = 0
    g_sumloss = 0
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    with open(a.outfile + '{:03d}.txt'.format(epoch), mode='w', encoding='utf-8') as file_obj:
        with torch.no_grad():
            for step, (imgs, names) in pbar:
                imgs = imgs.cuda()
                outputs = model(imgs)
                outputs = outputs.cpu().numpy()
                mean = np.array([25, 25, 25, 0.3, 0.3, 0.3, 0], dtype=np.float32)
                std = np.array([275, 275, 275, 0.65, 0.65, 0.65, 1], dtype=np.float32)
                outputs = outputs * std
                outputs = outputs + mean
                outputs = str(outputs.tolist())
                file_obj.write('\npred:')
                file_obj.write(outputs)
                # del rgb_sumloss, sigma_sumloss
                torch.cuda.empty_cache()

        # file_obj.write( f'\nVAL epoch {epoch:03d} sigma_loss:{sigma_sumloss/(step + 1)} rgb_loss:{rgb_sumloss/(step + 1)}  g_loss:{g_sumloss/(step + 1)}')
    return sigma_sumloss/len(val_loader), rgb_sumloss/len(val_loader)

def init_net(net, init_type='xavier'):
    init_weights(net, init_type)
    return net


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

# @profile
def main():
    # torch.set_num_threads(1)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    val_loader = prepare_dataloader()
    torch.manual_seed(a.seed)
    if torch.cuda.is_available() and not a.no_cuda:
        print("Let's use ", torch.cuda.device_count(), " GPUs!")
    model = Generator()
    model = nn.DataParallel(model).cuda()
    # model_name = os.path.join(a.model_dir, 'model006.pth')
    # model.load_state_dict(torch.load(model_name))
    # init_weights(model, init_type='kaiming')
    # model = model.cuda()


    # logging init
    logging.basicConfig(filename=os.path.join('./', 'train.log'),
                        format='[%(asctime)s]%(message)s', level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    loadmodel_name = os.path.join(a.model_dir, 'model{:03d}.pth'.format(a.init_epoch))
    if os.path.exists(loadmodel_name):
        model.load_state_dict(torch.load(loadmodel_name))
        print(loadmodel_name + ' model has been loaded')

    if not os.path.exists(a.model_dir):
        os.makedirs(a.model_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.92)
    if a.mode == 'test':
        # if not os.path.exists(a.output_dir):
        #     os.makedirs(a.output_dir)
        test_epoch(a.init_epoch, model, val_loader)
        return

if __name__ == '__main__':
    main()