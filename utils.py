import torch.nn as nn
import numpy as np
import torch 
import config
import random
import os
import glob
import nibabel as nib

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar", filestage=None, **extra):
    # Keep backward compatibility with the older API that passed `filestage`.
    if filestage is not None:
        filename = filestage
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # Store any extra metadata (e.g., epoch, best metric) when provided.
    for key, value in extra.items():
        if value is not None:
            checkpoint[key] = value
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # Return checkpoint to allow callers to recover metadata when present.
    return checkpoint

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_latent_stats(latent_dir):
    """Compute mean/std over all latent SPECT volumes in `latent_dir`."""
    files = glob.glob(os.path.join(latent_dir, "*.nii")) + glob.glob(os.path.join(latent_dir, "*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"No latent files found in {latent_dir}")
    total = 0.0
    total_sq = 0.0
    count = 0
    for f in files:
        arr = nib.load(f).get_fdata().astype(np.float64)
        total += arr.sum()
        total_sq += np.square(arr).sum()
        count += arr.size
    mean = total / count
    var = total_sq / count - mean * mean
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std
