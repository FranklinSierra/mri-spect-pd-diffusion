import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import TwoDataset
from main import Diffusion
from model import UNet
from utils import compute_latent_stats, seed_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genera latentes sintetizados con un UNet difusor ya entrenado "
            "y guarda los embeddings (más sus etiquetas) para usarlos en un clasificador."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="result/exp_cond_unet_min_max/Unet.pth.tar",
        help="Ruta al checkpoint del UNet (por defecto usa la carpeta solicitada).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result/cond_unet_latents",
        help="Directorio donde se almacenarán los .npy y el CSV con metadatos.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        choices=["train", "validation", "test"],
        help="Conjuntos sobre los que se extraerán latentes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Tamaño de batch para el DataLoader (usa el mismo valor de entrenamiento por defecto).",
    )
    return parser.parse_args()


def build_dataloader(split: str, batch_size: int) -> DataLoader:
    task_file = {
        "train": config.train,
        "validation": config.validation,
        "test": config.test,
    }[split]
    dataset = TwoDataset(
        root_MRI=config.whole_MRI,
        root_Abeta=config.latent_Abeta,
        task=task_file,
        stage=split,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.numworker,
        pin_memory=True,
        drop_last=False,
    )


def load_unet(checkpoint_path: str) -> torch.nn.Module:
    unet = UNet(
        in_channel=2,
        out_channel=1,
        image_size=config.latent_shape[0],
    ).to(config.device)
    state = torch.load(checkpoint_path, map_location=config.device)
    state_dict = state.get("state_dict", state)
    try:
        unet.load_state_dict(state_dict)
    except RuntimeError:
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                cleaned[key.replace("module.", "", 1)] = value
            else:
                cleaned[key] = value
        unet.load_state_dict(cleaned)
    unet.eval()
    return unet


@torch.no_grad()
def diffuse_latents(
    model: torch.nn.Module,
    diffusion: Diffusion,
    cond: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    n = cond.shape[0]
    device = cond.device
    latents = torch.randn((n, 1, *config.latent_shape), device=device)
    for i in reversed(range(1, diffusion.noise_steps)):
        t = torch.full((n,), i, dtype=torch.long, device=device)
        pred_noise = model(latents, cond, t, labels)
        alpha = diffusion.alpha[t][:, None, None, None, None]
        alpha_hat = diffusion.alpha_hat[t][:, None, None, None, None]
        beta = diffusion.beta[t][:, None, None, None, None]
        if i > 1:
            noise = torch.randn_like(latents)
        else:
            noise = torch.zeros_like(latents)
        latents = (
            1
            / torch.sqrt(alpha)
            * (
                latents
                - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise
            )
            + torch.sqrt(beta) * noise
        )
    return latents


def save_embeddings(
    split: str,
    latent_batch: np.ndarray,
    names: List[str],
    labels: np.ndarray,
    out_dir: str,
) -> List[Tuple[str, int, str]]:
    rows = []
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    for latent, name, label in zip(latent_batch, names, labels):
        base = os.path.splitext(name)[0]
        out_path = os.path.join(split_dir, f"{base}.npy")
        np.save(out_path, latent.astype(np.float32))
        rows.append((base, int(label), out_path))
    return rows


def main():
    args = parse_args()
    seed_torch(config.seed)
    diffusion = Diffusion()
    unet = load_unet(args.checkpoint)
    latent_mean, latent_std = compute_latent_stats(config.latent_Abeta)
    latent_mean_t = torch.tensor(latent_mean, device=config.device).view(
        1, 1, 1, 1, 1
    )
    latent_std_t = torch.tensor(latent_std, device=config.device).view(
        1, 1, 1, 1, 1
    )
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        loader = build_dataloader(split, args.batch_size)
        metadata: List[Tuple[str, int, str]] = []
        for MRI, _, _, names, label in loader:
            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI, device=config.device, dtype=torch.float32)
            label_tensor = torch.tensor(label, device=config.device).long()
            latents = diffuse_latents(unet, diffusion, MRI, label_tensor)
            latents = latents * latent_std_t + latent_mean_t
            latents_np = latents.cpu().numpy().squeeze(1)
            metadata.extend(
                save_embeddings(
                    split,
                    latents_np,
                    names,
                    label_tensor.cpu().numpy().reshape(-1),
                    args.output_dir,
                )
            )
        meta_path = os.path.join(args.output_dir, f"{split}_metadata.csv")
        with open(meta_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "latent_path"])
            writer.writerows(metadata)
        print(f"[{split}] Guardados {len(metadata)} latentes en {args.output_dir}")


if __name__ == "__main__":
    main()
