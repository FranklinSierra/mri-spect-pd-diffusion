import torch, nibabel as nib, os
from torch.utils.data import DataLoader
from model import AAE, UNet
from utils import load_checkpoint
from dataset import TwoDataset
from main import Diffusion
import config

device = config.device
aae = AAE().to(device)
opt_aae = torch.optim.Adam(aae.parameters(), lr=config.learning_rate)
load_checkpoint(config.CHECKPOINT_AAE, aae, opt_aae, config.learning_rate)
aae.eval()

unet = UNet().to(device)
opt_unet = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
unet = torch.nn.DataParallel(unet, device_ids=config.gpus, output_device=config.gpus[0])
load_checkpoint(config.CHECKPOINT_Unet, unet, opt_unet, config.learning_rate)
unet.eval()

dataset = TwoDataset(root_MRI=config.whole_MRI,
                     root_Abeta=config.whole_Abeta,
                     task=config.test, stage="test")
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.numworker)
diffusion = Diffusion()

with torch.no_grad():
    for MRI, _, names, labels in loader:
        MRI = MRI.unsqueeze(1).to(device)
        sampled_latent = diffusion.sample(unet, MRI)
        synth = aae.decoder(sampled_latent).clamp(0, 1).cpu().squeeze().numpy()
        affine = nib.load(os.path.join(config.whole_MRI, names[0])).affine
        nib.save(nib.Nifti1Image(synth, affine),
                 os.path.join("result", config.exp, "samples", names[0]))
