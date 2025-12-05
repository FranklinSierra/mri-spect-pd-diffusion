import torch, nibabel as nib, os
from torch.utils.data import DataLoader, Dataset
from model import AAE, UNet
from utils import load_checkpoint
from dataset import resolve_nifti, center_crop, z_score_norm
from main import Diffusion
import config
import numpy as np

class MRIDataset(Dataset):
    """MRI-only dataset for inference using IDs from the specified split."""
    def __init__(self, ids_file, root_mri):
        self.ids = [i.strip() for i in open(ids_file) if i.strip()]
        self.root_mri = root_mri

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        bid = self.ids[idx]
        path_mri = resolve_nifti(self.root_mri, bid)
        mri = nib.load(path_mri).get_fdata().astype(np.float32)
        mri = center_crop(mri, config.crop_size)
        mri = z_score_norm(mri)
        return torch.tensor(mri, dtype=torch.float32), bid + ".nii"

device = config.device
aae = AAE().to(device)
opt_aae = torch.optim.Adam(aae.parameters(), lr=config.learning_rate)
load_checkpoint(config.CHECKPOINT_AAE, aae, opt_aae, config.learning_rate)
aae.eval()

unet = UNet(in_channel=2, out_channel=1, image_size=config.latent_shape[0]).to(device)
opt_unet = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
unet = torch.nn.DataParallel(unet, device_ids=config.gpus, output_device=config.gpus[0])
load_checkpoint(config.CHECKPOINT_Unet, unet, opt_unet, config.learning_rate)
unet.eval()

dataset = MRIDataset(ids_file=config.train, root_mri=config.whole_MRI)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.numworker)
diffusion = Diffusion()
os.makedirs(os.path.join("result", str(config.exp), "samples"), exist_ok=True)

with torch.no_grad():
    for MRI, names in loader:
        MRI = MRI.unsqueeze(1).to(device)
        sampled_latent = diffusion.sample(unet, MRI)
        synth = aae.decoder(sampled_latent).clamp(0, 1).cpu().squeeze().numpy()
        affine = nib.load(os.path.join(config.whole_MRI, names[0])).affine
        nib.save(nib.Nifti1Image(synth, affine),
                 os.path.join("result", config.exp, "samples", names[0]))
