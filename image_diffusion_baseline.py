import os
import copy
import csv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import config
from dataset import resolve_nifti, nifti_to_numpy, min_max_norm, z_score_norm, crop
from model import UNet
from utils import seed_torch


class ImagePairDataset(Dataset):
    """Direct MRI-SPECT pairs (full image) with basic normalization."""

    def __init__(self, ids_txt, root_mri, root_spect, stage="train"):
        self.ids = [i.strip() for i in open(ids_txt) if i.strip()]
        self.root_mri = root_mri
        self.root_spect = root_spect
        self.stage = stage
        import pandas as pd
        self.labels = pd.read_csv("data_info/data_info.csv", encoding="ISO-8859-1")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        bid = self.ids[idx % len(self.ids)]
        mri_path = resolve_nifti(self.root_mri, bid)
        spect_path = resolve_nifti(self.root_spect, bid)

        mri = crop(z_score_norm(nifti_to_numpy(mri_path)))
        spect = crop(min_max_norm(nifti_to_numpy(spect_path)))

        mri = torch.tensor(mri[None, ...], dtype=torch.float32)
        spect = torch.tensor(spect[None, ...], dtype=torch.float32)

        label = self.labels[self.labels["ID"].astype(str) == bid]["label"].values.astype(
            np.float32
        )
        if label.size == 0:
            label = np.array([0], dtype=np.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return mri, spect, bid, label


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.device = device
        self.noise_steps = noise_steps
        self.beta = torch.linspace(beta_start, beta_end, noise_steps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,), device=self.device)

    @torch.no_grad()
    def sample(self, model, y, labels=None):
        model.eval()
        n = y.shape[0]
        x = torch.randn((n, 1, 40, 48, 40), device=y.device)
        for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps - 1, leave=False):
            t = torch.full((n,), i, device=y.device, dtype=torch.long)
            pred = model(x, y, t, labels)
            alpha = self.alpha[t][:, None, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None, None]
            beta = self.beta[t][:, None, None, None, None]
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * pred
            ) + torch.sqrt(beta) * noise
        return x


def safe_ssim(gt, pred, data_range):
    min_side = min(gt.shape)
    win = 7 if min_side >= 7 else (min_side if min_side % 2 == 1 else max(min_side - 1, 1))
    if win < 3:
        return 0.0
    return ssim(gt, pred, data_range=data_range, win_size=win)


def train_image_diffusion(
    epochs=300,
    batch_size=4,
    lr=2e-4,
    noise_steps=1000,
    patience_limit=20,
    save_dir="result/exp_image_diffusion",
):
    os.makedirs(save_dir, exist_ok=True)
    seed_torch(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ImagePairDataset(config.train, config.whole_MRI, config.whole_Abeta, stage="train")
    val_ds = ImagePairDataset(config.validation, config.whole_MRI, config.whole_Abeta, stage="val")
    test_ds = ImagePairDataset(config.test, config.whole_MRI, config.whole_Abeta, stage="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=config.numworker, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=config.numworker, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=config.numworker, pin_memory=True
    )

    unet = UNet(in_channel=2, out_channel=1, image_size=40).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    diffusion = Diffusion(noise_steps=noise_steps, device=device)
    ema = copy.deepcopy(unet).eval().requires_grad_(False)
    mse = nn.MSELoss()

    best_ssim = -1e9
    patience = 0

    loss_csv = os.path.join(save_dir, "loss_curve.csv")
    val_csv = os.path.join(save_dir, "validation.csv")
    test_csv = os.path.join(save_dir, "test.csv")

    # Write headers
    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "MSE_loss"])
    if not os.path.exists(val_csv):
        with open(val_csv, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "PSNR", "SSIM"])
    if not os.path.exists(test_csv):
        with open(test_csv, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "PSNR", "SSIM"])

    for epoch in range(epochs):
        unet.train()
        loop = tqdm(train_loader, leave=False)
        epoch_loss = 0
        for mri, spect, bid, label in loop:
            mri, spect, label = mri.to(device), spect.to(device), label.to(device)
            spect_low = F.interpolate(spect, size=(40, 48, 40), mode="trilinear", align_corners=False)
            t = diffusion.sample_timesteps(spect_low.shape[0])
            x_t, noise = diffusion.noise_images(spect_low, t)
            pred_noise = unet(x_t, mri, t, label)
            loss = mse(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for p, q in zip(ema.parameters(), unet.parameters()):
                p.data = 0.999 * p.data + 0.001 * q.data

            epoch_loss += loss.item()
            loop.set_description(f"epoch {epoch+1} loss {loss.item():.4f}")

        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, epoch_loss / len(train_loader)])

        # Validation
        unet.eval()
        psnr_sum = 0
        ssim_sum = 0
        with torch.no_grad():
            for mri, spect, bid, label in val_loader:
                mri, spect, label = mri.to(device), spect.to(device), label.to(device)
                spect_low = F.interpolate(
                    spect, size=(40, 48, 40), mode="trilinear", align_corners=False
                )
                sampled = diffusion.sample(ema, mri, label)
                # Upscale recon to original crop size for metrics
                recon_up = F.interpolate(sampled, size=spect.shape[2:], mode="trilinear", align_corners=False)

                recon = recon_up.cpu().numpy().squeeze().astype(np.float32)
                target = spect.cpu().numpy().squeeze().astype(np.float32)
                data_range = max(target.max() - target.min(), 1e-8)
                psnr_sum += psnr(target, recon, data_range=data_range)
                ssim_sum += safe_ssim(target, recon, data_range=data_range)

        psnr_avg = psnr_sum / max(len(val_loader), 1)
        ssim_avg = ssim_sum / max(len(val_loader), 1)
        with open(val_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, psnr_avg, ssim_avg])
        print(f"Epoch {epoch+1}: loss {epoch_loss/len(train_loader):.4f}, PSNR {psnr_avg:.3f}, SSIM {ssim_avg:.4f}")

        if ssim_avg > best_ssim:
            best_ssim = ssim_avg
            patience = 0
            torch.save(
                {
                    "state_dict": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "psnr": psnr_avg,
                    "ssim": ssim_avg,
                },
                os.path.join(save_dir, "unet_image_best.pth"),
            )
            # Evaluate on test split when improving
            psnr_sum = 0
            ssim_sum = 0
            with torch.no_grad():
                for mri, spect, bid, label in test_loader:
                    mri, spect, label = mri.to(device), spect.to(device), label.to(device)
                    sampled = diffusion.sample(ema, mri, label)
                    recon_up = F.interpolate(
                        sampled, size=spect.shape[2:], mode="trilinear", align_corners=False
                    )
                    recon = recon_up.cpu().numpy().squeeze().astype(np.float32)
                    target = spect.cpu().numpy().squeeze().astype(np.float32)
                    data_range = max(target.max() - target.min(), 1e-8)
                    psnr_sum += psnr(target, recon, data_range=data_range)
                    ssim_sum += safe_ssim(target, recon, data_range=data_range)
            psnr_test = psnr_sum / max(len(test_loader), 1)
            ssim_test = ssim_sum / max(len(test_loader), 1)
            with open(test_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch + 1, psnr_test, ssim_test])
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping en epoch {epoch+1}")
                break


if __name__ == "__main__":
    train_image_diffusion()
