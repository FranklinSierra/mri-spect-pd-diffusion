import argparse
import csv
import os
from typing import List, Tuple
from glob import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import (
    TwoDataset,
    center_crop,
    min_max_norm,
    read_list,
    resolve_nifti,
    resample_to_shape,
)
from main import Diffusion
from model import AAE, UNet
from utils import compute_latent_stats, seed_torch


def get_task_file(split: str) -> str:
    return {
        "train": config.train,
        "validation": config.validation,
        "test": config.test,
    }.get(split, "")


def get_latent_root(split: str) -> str:
    if split == "test":
        return os.path.join(config.latent_Abeta, "test")
    return config.latent_Abeta


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
        choices=["train", "validation", "test", "prodromal"],
        help="Conjuntos sobre los que se extraerán latentes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Tamaño de batch para el DataLoader (usa el mismo valor de entrenamiento por defecto).",
    )
    parser.add_argument(
        "--noise_steps",
        type=int,
        default=1000,
        help="Número de pasos de la cadena de difusión (por defecto 1000).",
    )
    return parser.parse_args()


def build_dataloader(split: str, batch_size: int) -> DataLoader:
    if split == "prodromal":
        dataset = ProdromalMRIDataset(root_MRI="data/prodromal_MRI")
    else:
        task_file = get_task_file(split)
        latent_root = get_latent_root(split)
        dataset = TwoDataset(
            root_MRI=config.whole_MRI,
            root_Abeta=latent_root,
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


_AAE_ENCODER = None


def get_aae_encoder() -> torch.nn.Module:
    global _AAE_ENCODER
    if _AAE_ENCODER is None:
        if not os.path.exists(config.CHECKPOINT_AAE):
            raise FileNotFoundError(
                f"No se encontró el checkpoint del AAE en {config.CHECKPOINT_AAE}"
            )
        print("Cargando encoder del AAE congelado...")
        aae = AAE().to(config.device)
        checkpoint = torch.load(config.CHECKPOINT_AAE, map_location=config.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        aae.load_state_dict(state_dict)
        aae.eval()
        for param in aae.parameters():
            param.requires_grad = False
        _AAE_ENCODER = aae.encoder
    return _AAE_ENCODER


def latent_exists(latent_root: str, basename: str) -> bool:
    nii_path = os.path.join(latent_root, f"{basename}.nii")
    nii_gz_path = nii_path + ".gz"
    return os.path.exists(nii_path) or os.path.exists(nii_gz_path)


def strip_nii_extension(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


class ProdromalMRIDataset(torch.utils.data.Dataset):
    """Dataset para inferencia solo con MRI prodromal (sin SPECT ni labels)."""

    def __init__(self, root_MRI: str = "data/prodromal_MRI"):
        self.root_MRI = root_MRI
        self.paths = sorted(glob(os.path.join(root_MRI, "*.nii*")))
        if not self.paths:
            raise FileNotFoundError(f"No se encontraron NIfTI en {root_MRI}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.basename(path)
        basename = strip_nii_extension(name)
        img = nib.load(path)
        MRI = img.get_fdata().astype(np.float32)
        if not np.isfinite(MRI).all():
            finite = np.isfinite(MRI)
            MRI = np.where(finite, MRI, MRI[finite].min() if finite.any() else 0.0)
        MRI = resample_to_shape(MRI, config.target_shape)
        MRI = center_crop(MRI, config.crop_size)
        MRI = min_max_norm(MRI)
        # Usamos 0 como etiqueta dummy para ser consistente con el split de test (control/parkinson).
        label = np.array([0], dtype=np.float32)
        # Devolvemos placeholders para Abeta/Abeta_real para mantener compatibilidad.
        return MRI, MRI, MRI, basename + ".nii.gz", label


def encode_and_store_latent(basename: str, encoder: torch.nn.Module, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    src_path = resolve_nifti(config.whole_Abeta, basename)
    img = nib.load(src_path)
    vol = img.get_fdata().astype(np.float32)
    if not np.isfinite(vol).all():
        finite_mask = np.isfinite(vol)
        replacement = vol[finite_mask].min() if finite_mask.any() else 0.0
        vol = np.where(finite_mask, vol, replacement)
    vol = resample_to_shape(vol, config.target_shape)
    vol = center_crop(vol, config.crop_size)
    vol = min_max_norm(vol)
    vol = torch.tensor(
        vol[None, None, ...], dtype=torch.float32, device=config.device
    )
    with torch.no_grad():
        latent = encoder(vol).cpu().numpy().squeeze().astype(np.float32)
    out_path = os.path.join(out_dir, f"{basename}.nii")
    nib.save(nib.Nifti1Image(latent, img.affine), out_path)


def ensure_latents_for_split(split: str, latent_root: str, task_file: str):
    if split != "test":
        os.makedirs(latent_root, exist_ok=True)
        return
    ids = read_list(task_file)
    missing = [x for x in ids if not latent_exists(latent_root, x)]
    if not missing:
        return
    encoder = get_aae_encoder()
    print(
        f"[{split}] Generando {len(missing)} latentes faltantes con el encoder AAE..."
    )
    for basename in missing:
        encode_and_store_latent(basename, encoder, latent_root)


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
        base = strip_nii_extension(name)
        out_path = os.path.join(split_dir, f"{base}.npy")
        np.save(out_path, latent.astype(np.float32))
        rows.append((base, int(label), out_path))
    return rows


def main():
    args = parse_args()
    print("Configuración de extracción de latentes:", args)
    seed_torch(config.seed)
    diffusion = Diffusion(noise_steps=args.noise_steps)
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
        task_file = get_task_file(split)
        latent_root = get_latent_root(split)
        if split != "prodromal":
            ensure_latents_for_split(split, latent_root, task_file)
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        meta_path = os.path.join(args.output_dir, f"{split}_metadata.csv")
        metadata: List[Tuple[str, int, str]] = []
        existing_ids = set()
        if os.path.exists(meta_path):
            with open(meta_path, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        id_, label_str, path = row[:3]
                        metadata.append((id_, int(label_str), path))
                        existing_ids.add(id_)
        for fname in os.listdir(split_dir):
            if fname.endswith(".npy"):
                existing_ids.add(strip_nii_extension(fname))
        loader = build_dataloader(split, args.batch_size)
        for MRI, _, _, names, label in loader:
            keep_idx = [
                idx
                for idx, name in enumerate(names)
                if strip_nii_extension(name) not in existing_ids
            ]
            if not keep_idx:
                continue
            MRI = MRI[keep_idx]
            names = [names[i] for i in keep_idx]
            label = label[keep_idx]
            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI, device=config.device, dtype=torch.float32)
            # mantener las etiquetas originales para el CSV, pero NO usarlas
            # como condición al generar latentes: pasar un tensor de ceros
            label_tensor = torch.tensor(label, device=config.device).long()
            zero_labels = torch.zeros_like(label_tensor)
            latents = diffuse_latents(unet, diffusion, MRI, zero_labels)
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
            existing_ids.update(strip_nii_extension(name) for name in names)
        with open(meta_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "latent_path"])
            writer.writerows(metadata)
        print(f"[{split}] Guardados {len(metadata)} latentes en {args.output_dir}")


if __name__ == "__main__":
    main()
