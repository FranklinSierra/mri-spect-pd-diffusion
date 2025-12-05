import os
import csv
import numpy as np
import nibabel as nib
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import config
from dataset import resolve_nifti, center_crop
from model import AAE
from utils import seed_torch


def main():
    seed_torch(config.seed)
    device = torch.device(config.device)
    out_dir = os.path.join("result", str(config.exp), "aae_recon_test")
    os.makedirs(out_dir, exist_ok=True)

    # Cargar modelo AAE
    model = AAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ckpt = torch.load(config.CHECKPOINT_AAE, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # IDs de test
    ids = [i.strip() for i in open(config.test) if i.strip()]

    psnr_sum = 0.0
    ssim_sum = 0.0
    metrics_path = os.path.join(out_dir, "aae_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "PSNR", "SSIM"])

        for bid in ids:
            path = resolve_nifti(config.whole_Abeta, bid)
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32)
            data = center_crop(data, config.crop_size)
            vmin, vmax = data.min(), data.max()
            data = np.zeros_like(data) if vmax - vmin < 1e-8 else (data - vmin) / (vmax - vmin)

            x = torch.tensor(data[None, None, ...], device=device)
            with torch.no_grad():
                recon = model(x)
            recon = torch.clamp(recon, 0, 1).cpu().numpy().squeeze().astype(np.float32)

            if recon.shape != data.shape:
                print(f"{bid}: shape mismatch data {data.shape} recon {recon.shape}, skipping")
                continue

            rng = max(data.max() - data.min(), 1e-8)
            ps = psnr(data, recon, data_range=rng)
            ss = ssim(data, recon, data_range=rng)
            writer.writerow([bid, ps, ss])
            psnr_sum += ps
            ssim_sum += ss

            # Guardar reconstrucción
            nib.save(nib.Nifti1Image(recon, img.affine), os.path.join(out_dir, f"{bid}_recon.nii.gz"))

    if ids:
        print(f"Promedio PSNR {psnr_sum/len(ids):.3f}, SSIM {ssim_sum/len(ids):.4f}")
    else:
        print("No hay IDs en el split de test.")


if __name__ == "__main__":
    main()
