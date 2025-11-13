import argparse
import os
import random
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import math


class SRCNN(nn.Module):
    def __init__(self, channels=1):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=64,
            kernel_size=9,
            padding=9 // 2
        )

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=1,
            padding=0
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=channels,
            kernel_size=5,
            padding=5 // 2
        )

        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module is self.conv3:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias.data)
                else:
                    try:
                        kernel_area = module.weight.data[0][0].numel()
                        std = math.sqrt(2.0 / (module.out_channels * kernel_area))
                    except Exception:
                        std = 0.01
                    nn.init.normal_(module.weight.data, 0.0, std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias.data)

    def forward(self, x):
        # --- Layer 1 ---
        x = self.conv1(x)
        x = self.relu(x)

        # --- Layer 2 ---
        x = self.conv2(x)
        x = self.relu(x)

        # --- Layer 3 ---
        x = self.conv3(x)

        return x


class PairedDIV2KDataset(Dataset):

    def __init__(self, hr_dir, lr_dir, scale=8, patch_size=128, mode='train'):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.mode = mode

        hr_files = [p.name for p in self.hr_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        self.pairs = []
        for hr_name in sorted(hr_files):
            stem = Path(hr_name).stem
            lr_name = f"{stem}x{scale}.png"
            lr_path = self.lr_dir / lr_name
            if ':' in lr_name:
                continue
            if lr_path.exists():
                self.pairs.append((self.hr_dir / hr_name, lr_path))

        if len(self.pairs) == 0:
            raise RuntimeError(f"No pairs found in {hr_dir} and {lr_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hr_path, lr_path = self.pairs[idx]
        hr = Image.open(hr_path).convert('YCbCr')
        lr = Image.open(lr_path).convert('YCbCr')

        lr_up = lr.resize(hr.size, resample=Image.BICUBIC)

        w, h = hr.size
        ps = min(self.patch_size, w, h)

        if self.mode == 'train':
            if w == ps and h == ps:
                left = 0
                top = 0
            else:
                left = random.randint(0, w - ps)
                top = random.randint(0, h - ps)
        else:
            left = max(0, (w - ps) // 2)
            top = max(0, (h - ps) // 2)

        hr_patch = hr.crop((left, top, left + ps, top + ps))
        lr_patch = lr_up.crop((left, top, left + ps, top + ps))

        if self.mode == 'train' and random.random() > 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)

        hr_y = hr_patch.split()[0]
        lr_y = lr_patch.split()[0]

        hr_t = self.to_tensor(hr_y)
        lr_t = self.to_tensor(lr_y)

        return lr_t, hr_t


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    dataset = PairedDIV2KDataset(args.hr_dir, args.lr_dir, scale=args.scale, patch_size=args.patch_size, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = None
    best_psnr = -1.0
    if args.val_hr_dir and args.val_lr_dir:
        try:
            val_dataset = PairedDIV2KDataset(args.val_hr_dir, args.val_lr_dir, scale=args.scale, patch_size=args.patch_size, mode='val')
            val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size or 1, shuffle=False, num_workers=args.num_workers)
            print(f"Validation dataset: {len(val_dataset)} images")
        except Exception as e:
            print(f"Warning: could not create validation dataset: {e}")

    model = SRCNN(channels=1).to(device)
    criterion = torch.nn.MSELoss()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    print(f"Using optimizer: {args.optimizer}, lr={args.lr}")

    scheduler = None
    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr
        )

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        for lr_batch, hr_batch in pbar:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            out = model(lr_batch)
            loss = criterion(out, hr_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss * lr_batch.size(0)
            pbar.set_postfix(loss=batch_loss)

        epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch} average loss: {epoch_loss:.6f}")

        if val_loader is not None and (args.val_freq > 0 and epoch % args.val_freq == 0):
            model.eval()
            running_psnr = 0.0
            running_n = 0
            with torch.no_grad():
                for lr_batch, hr_batch in tqdm(val_loader, desc=f"Validation epoch {epoch}"):
                    lr_batch = lr_batch.to(device)
                    hr_batch = hr_batch.to(device)

                    out = model(lr_batch).clamp(0.0, 1.0)

                    crop = args.scale
                    _, _, h, w = out.shape
                    if h > 2 * crop and w > 2 * crop:
                        out_c = out[:, :, crop:h - crop, crop:w - crop]
                        hr_c = hr_batch[:, :, crop:h - crop, crop:w - crop]
                    else:
                        out_c = out
                        hr_c = hr_batch

                    mse = torch.mean((out_c - hr_c) ** 2, dim=[1, 2, 3])
                    psnr_vals = 10.0 * torch.log10((1.0 ** 2) / (mse + 1e-10))
                    running_psnr += psnr_vals.sum().item()
                    running_n += psnr_vals.numel()

            if running_n > 0:
                avg_psnr = running_psnr / running_n
                print(f"Validation PSNR (epoch {epoch}): {avg_psnr:.4f} dB")
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_path = os.path.join(args.save_dir, 'best_srcnn.pt')
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, best_path)
                    print(f"Saved best model (PSNR={best_psnr:.4f}) to: {best_path}")

        if scheduler is not None:
            if args.scheduler_type == 'step':
                scheduler.step()
            elif args.scheduler_type == 'plateau':
                if avg_psnr is not None:
                    scheduler.step(avg_psnr)

        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Current LRs: {lrs}")


def parse_args():
    p = argparse.ArgumentParser(description='Train SRCNN on DIV2K pairs')
    p.add_argument('--hr-dir', type=str, default='images/DIV2K_train_HR', help='Path to HR images')
    p.add_argument('--lr-dir', type=str, default='images/DIV2K_train_LR_bicubic/X4', help='Path to LR images')
    p.add_argument('--scale', type=int, default=4, help='Upscale factor in LR filenames (x8)')
    p.add_argument('--patch-size', type=int, default=12, help='HR patch size for training')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    # Optimizer options
    p.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer to use for training')
    p.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (if optimizer=sgd)')
    p.add_argument('--weight-decay', type=float, default=1e-7, help='Weight decay (L2)')
    p.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Adam betas as two floats')
    p.add_argument('--save-dir', type=str, default='output/checkpoints', help='Where to save checkpoints')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--cpu', action='store_true', help='Force CPU')
    # validation options
    p.add_argument('--val-hr-dir', type=str, default=None, help='Path to HR validation images')
    p.add_argument('--val-lr-dir', type=str, default=None, help='Path to LR validation images')
    p.add_argument('--val-freq', type=int, default=1, help='Run validation every N epochs (0 to disable)')
    p.add_argument('--val-batch-size', type=int, default=1, help='Batch size for validation')
    # Inference / upscaling options
    p.add_argument('--infer', action='store_true', help='Run inference/upscale a single image instead of training')
    p.add_argument('--input', type=str, default=None, help='Path to input LR image for inference')
    p.add_argument('--output', type=str, default='output/upscaled.png', help='Path to save upscaled image')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (.pt) to load for inference')
    p.add_argument('--out-scale', type=int, default=None, help='Scale factor to upsample input before model (if omitted, uses --scale)')
    p.add_argument('--gt', type=str, default=None, help='Path to ground-truth HR image to compute PSNR after inference')
    # scheduler options
    p.add_argument('--scheduler-type', type=str, default='plateau', choices=['none', 'step', 'plateau'],
                   help='Type of LR scheduler to use')
    p.add_argument('--step-size', type=int, default=10, help='StepLR step size (epochs)')
    p.add_argument('--gamma', type=float, default=0.5, help='StepLR gamma')
    p.add_argument('--patience', type=int, default=5, help='ReduceLROnPlateau patience (epochs)')
    p.add_argument('--factor', type=float, default=0.5, help='ReduceLROnPlateau factor')
    p.add_argument('--min-lr', type=float, default=1e-7, help='Minimum LR for schedulers')
    return p.parse_args()


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    if args.input is None:
        raise ValueError('Please provide --input path for inference')

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        save_dir = Path(args.save_dir)
        if save_dir.exists():
            ckpts = sorted(save_dir.glob('srcnn_epoch*.pt'))
            if ckpts:
                ckpt_path = str(ckpts[-1])

    if ckpt_path is None:
        raise ValueError('No checkpoint provided and none found in save-dir')

    model = SRCNN(channels=1).to(device)
    map_location = 'cpu' if args.cpu or not torch.cuda.is_available() else None
    ckpt = torch.load(ckpt_path, map_location=map_location)

    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    model.load_state_dict(state_dict)
    model.eval()


    inp = Image.open(args.input).convert('RGB')
    w, h = inp.size
    scale = args.out_scale if args.out_scale is not None else args.scale
    out_size = (w * scale, h * scale)
    inp_up = inp.resize(out_size, resample=Image.BICUBIC)

    inp_up_ycbcr = inp_up.convert('YCbCr')
    y, cb, cr = inp_up_ycbcr.split()

    to_tensor = transforms.ToTensor()
    inp_t = to_tensor(y).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_t = model(inp_t)

    sr_t = sr_t.squeeze(0).cpu().clamp(0.0, 1.0)
    to_pil = transforms.ToPILImage()
    sr_y_pil = to_pil(sr_t)

    sr_img = Image.merge('YCbCr', (sr_y_pil, cb, cr)).convert('RGB')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sr_img.save(str(out_path))
    print(f'Wrote upscaled image to: {out_path}')

    if getattr(args, 'gt', None):
        try:
            hr_img = Image.open(args.gt).convert('RGB')
            if hr_img.size != sr_img.size:
                print(f"Ground-truth image size {hr_img.size} differs from model output size {sr_img.size}; resizing ground-truth to match output for PSNR calculation.")
                hr_img = hr_img.resize(sr_img.size, resample=Image.BICUBIC)


            hr_y = hr_img.convert('YCbCr').split()[0]
            hr_t = to_tensor(hr_y)
            crop = args.scale

            _, h_hr, w_hr = hr_t.shape
            _, h_sr, w_sr = sr_t.shape

            if h_sr > 2 * crop and w_sr > 2 * crop and h_hr > 2 * crop and w_hr > 2 * crop:
                hr_tc = hr_t[:, crop:h_hr - crop, crop:w_hr - crop]
                sr_tc = sr_t[:, crop:h_sr - crop, crop:w_sr - crop]
            else:
                hr_tc = hr_t
                sr_tc = sr_t

            mse = torch.mean((sr_tc - hr_tc) ** 2).item()
            psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
            print(f"PSNR vs ground-truth (Y) ({args.gt}): {psnr:.4f} dB")
        except Exception as e:
            print(f"Could not compute PSNR for ground-truth {args.gt}: {e}")


if __name__ == '__main__':
    args = parse_args()
    if args.infer:
        print('Running inference with settings:', args)
        infer(args)
    else:
        print('Training with settings:', args)
        train(args)
