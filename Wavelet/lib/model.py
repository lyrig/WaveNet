from sympy import I
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import pywt
from tqdm import tqdm
import os
import sys
# from DPM.lib.cond_unet import Upsample
sys.path.append('/home/vision/diska4/shy/NerfDiff/Wavelet/lib')
from tensorboardX import SummaryWriter


from util import *



class WaveDataset(Dataset):
    def __init__(self, img_path, device='cpu') -> None:
        super().__init__()
        self.img_path = img_path
        # self.transform_path = transform_path
        self.imglist = sorted(os.listdir(img_path))
        self.device = device


    def __len__(self):
        return len(self.imglist)
    

    def __getitem__(self, index):
        imgR = read_img_asnp(os.path.join(self.img_path, self.imglist[index]))
        # img = imgR.copy()
        img = cv2.resize(imgR, None, fx=0.5, fy=0.5)
        img = cv2.resize(img, None, fx=2, fy=2)
        
        cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
        cA = torch.Tensor(cA)
        cH = torch.Tensor(cH)
        cV = torch.Tensor(cV)
        cD = torch.Tensor(cD)
        ret = torch.stack([cA, cH, cV, cD], dim=0).to(self.device)
        cA1, (cH1, cV1, cD1) = pywt.dwt2(imgR, 'haar')
        cA1 = torch.Tensor(cA1)
        cH1 = torch.Tensor(cH1)
        cV1 = torch.Tensor(cV1)
        cD1 = torch.Tensor(cD1)
        ret1 = torch.stack([cA1, cH1, cV1, cD1], dim=0).to(self.device)
        return imgR, img, ret, ret1

    

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.main = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x, target):
        B, C, H, W = x.shape
        hx = self.group_norm(x)
        hy = self.group_norm(target)
        q = self.proj_q(hx)
        k = self.proj_k(hy)
        v = self.proj_v(hy)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return h


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h



class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    """

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, channels, act='gelu', dropout=0.1) -> None:
        super().__init__()
        self.ln = LayerNorm(channels)
        self.fc1 = nn.Linear(channels, 2 * channels)
        self.fc2 = nn.Linear(2 * channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU(inplace=True)
        if act == 'gelu':
            self.activate = nn.GELU()
        elif act == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        x = inputs
        x = self.ln(x)
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + inputs


class WaveNet(nn.Module):
    def __init__(self, dim=64, in_channel=4, dim_mults=[1, 2, 4, 8]):
        super().__init__()
        init_dim = dim
        # block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.init_conv = nn.Conv2d(in_channel, init_dim, 7, padding = 3)
        num_resolutions = len(in_out)
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in),
                ResBlock(dim_in, dim_in),
                AttnBlock(dim_in),
                DownSample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        self.mid = nn.ModuleList([])
        mid_dim = dims[-1]
        self.mid.append(ResBlock(mid_dim, mid_dim))
        self.mid.append(AttnBlock(mid_dim))     
        self.mid.append(ResBlock(mid_dim, mid_dim))

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResBlock(dim_out + dim_in, dim_out),
                ResBlock(dim_out + dim_in, dim_out),
                AttnBlock(dim_out),
                UpSample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.final = ResBlock(dim * 2, dim)
        self.f = nn.Conv2d(dim, in_channel, 1)

    def forward(self, inp):
        x = self.init_conv(inp)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)
            
            x = block2(x)
            x = attn(x)
            h.append(x)
            # print(x.shape)
            # print(x.shape)

            x = downsample(x)

        block1, attn, block2 = self.mid
        x = block1(x)
        x = attn(x)
        x = block2(x)

        for block1, block2, attn, upsample in self.ups:
            
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final(x)
        return self.f(x)

class Trainer:
    def __init__(self,
                 model,
                 img_dir='/home/vision/diska4/shy/NerfDiff/data/LIDC/CTslice',
                 save_dir='/home/vision/diska4/shy/NerfDiff/Wavelet/result',
                 batch_size=4,
                 learning_rate=1e-4,
                 eps=1e-6,
                 train_step = 100,
                 max_grad_norm=1,
                 device='cpu'
                 ) -> None:
        self.model = model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=eps)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.ds = WaveDataset(img_dir, self.device)
        self.dl = DataLoader(self.ds, batch_size=batch_size, shuffle=True)
        self.schedule = LambdaLR(self.opt, lr_lambda=lambda epoch:1/(epoch +1))
        self.scaler = GradScaler(enabled=True)
        self.save_dir = save_dir
        self.train_step = train_step

        self.writer = SummaryWriter(save_dir)
    def train(self):
        for _ in range(self.train_step):
            loop = tqdm(enumerate(self.dl), total=len(self.dl))
            loop.set_description(f'Epoch {_}')
            total_loss = 0.0
            for i, (imgR,img, inp, oup) in loop:
                # print(inp.shape)
                with autocast(enabled=True):
                    pred = self.model(inp)
                    loss = nn.MSELoss()(pred, oup)
                    if self.max_grad_norm is not None:
                        # self.scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) 
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad()
                    total_loss += loss.cpu().item()
                    print(f'loss:{loss.cpu().item()}')
                    # break
            self.schedule.step()
            self.writer.add_scalar(f'Train Loss', total_loss/len(self.dl), _)

            
            if _ % 10 == 0 and _ != 0:

                print('Sampling')
                self.save(str(_))
                self.sample(5)
                # exit(0)
        print(f'Finish.')

    def save(self, milestone):
        data = {
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'version': '__version__'
        }
        torch.save(data, os.path.join(self.save_dir, f'model-{milestone}.pt'))
    @torch.no_grad()
    def sample(self, num=1):
        step = 0
        self.dd = DataLoader(self.ds, 1, False)
        for _, (imgR, img, inp, oup) in enumerate(self.dd):
            if step >= num:
                break
            pred = self.model(inp).squeeze(0)
            rA, rH, rV, rD = torch.chunk(pred, 4, 0)

            rA = rA[0].cpu().numpy()
            rH = rH[0].cpu().numpy()
            rV = rV[0].cpu().numpy()
            rD = rD[0].cpu().numpy()
            x_sr = pywt.idwt2((rA, (rH, rV, rD)), 'haar')
            img=img.squeeze(0).numpy()
            imgR = imgR.squeeze(0).numpy()
            result = x_sr + img
            # print(result[32])
            save_img_tensor(os.path.join(self.save_dir, f'pred_{_}.png'), result)
            save_img_tensor(os.path.join(self.save_dir, f'GT_{_}.png'), imgR)
            # exit(0)
            step += 1


if __name__ == '__main__':
    img_dir = '/home/vision/diska4/shy/NerfDiff/data/LIDC/CTslice'


    net = WaveNet()
    trainer = Trainer(net, device='cuda:2')
    trainer.train()
    # inp = torch.rand(size=(4, 4, 128, 128))
    # print(net(inp).shape)