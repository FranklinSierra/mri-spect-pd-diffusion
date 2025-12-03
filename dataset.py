from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import numpy as np
import config
import os

def read_list(file):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    return p

def nifti_to_numpy(file):
    data = nib.load(file).get_fdata()
    # Replace non-finite values with the minimum finite value in the volume
    # (or 0 if the whole volume is non-finite) instead of aborting.
    if not np.isfinite(data).all():
        finite_mask = np.isfinite(data)
        if finite_mask.any():
            min_finite = data[finite_mask].min()
        else:
            min_finite = 0.0
        data = np.where(finite_mask, data, min_finite)
    return data.astype(np.float32)

def resolve_nifti(root, basename):
    """Return a valid NIfTI path trying .nii first, then .nii.gz."""
    path_nii = os.path.join(root, basename + ".nii")
    if os.path.exists(path_nii):
        return path_nii
    path_niigz = os.path.join(root, basename + ".nii.gz")
    if os.path.exists(path_niigz):
        return path_niigz
    raise FileNotFoundError(f"NIfTI not found: {basename} (.nii or .nii.gz) under {root}")

def random_translation(data1):
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    return data1[10+i:170+i,18+j:210+j,10+z:170+z]

def crop(data1):
    return data1[10:170,18:210,10:170]

def min_max_norm(vol):
    vmin = vol.min()
    vmax = vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return (vol - vmin) / (vmax - vmin)

def z_score_norm(vol):
    mean = vol.mean()
    std = vol.std()
    if std < 1e-8:
        return np.zeros_like(vol)
    return (vol - mean) / std

# This is for the training of the first stage
class OneDataset(Dataset):
    def __init__(self, root_Abeta = config.whole_Abeta, task = config.train, stage = "train"):
        self.root_Abeta = root_Abeta
        self.task = task
        self.images = read_list(self.task)
        self.length_dataset = len(self.images)
        self.len = len(self.images)
        self.stage = stage

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        basename = self.images[index % self.len]
        path_Abeta = resolve_nifti(self.root_Abeta, basename)
        Abeta = nifti_to_numpy(path_Abeta)
        if self.stage == "train":
            Abeta = random_translation(Abeta)
        else:
            Abeta = crop(Abeta)
        Abeta = min_max_norm(Abeta)
        #print("min and max of Abeta:", Abeta.min(), Abeta.max())
        return Abeta, basename + ".nii"

# This is for the training of the second stage
class TwoDataset(Dataset):
    def __init__(self,root_MRI = config.whole_MRI, root_Abeta = config.whole_Abeta, task = config.train, stage = "train"):
        self.root_Abeta = root_Abeta
        self.root_MRI = root_MRI
        self.task = task
        self.images = read_list(self.task)
        self.length_dataset = len(self.images)
        self.len = len(self.images)
        self.stage = stage

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        basename = self.images[index % self.len]
        name = basename + ".nii"
        path_Abeta = resolve_nifti(self.root_Abeta, basename)
        Abeta = nifti_to_numpy(path_Abeta)
        path_MRI = resolve_nifti(self.root_MRI, basename)
        MRI = nifti_to_numpy(path_MRI)
        MRI = crop(MRI)
        MRI = z_score_norm(MRI)
        #print("min and max of MRI:", MRI.min(), MRI.max())
        if self.stage != "train":
            Abeta = crop(Abeta)
        Abeta = min_max_norm(Abeta)
        #print("min and max of Abeta:", Abeta.min(), Abeta.max())
        data = pd.read_csv("data_info/data_info.csv",encoding = "ISO-8859-1")
        label = data[data['ID'] == basename]['label'].values.astype(np.float32)
        # If label is missing, fall back to 0 to avoid shape mismatch later.
        if label.size == 0:
            label = np.array([0], dtype=np.float32)

        #print("ID", basename, "label shape", label.shape, "label", label)
        return MRI, Abeta, name, label
