import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
noiseSteps = 1000
latent_dim = 1
learning_rate = 2e-4
batch_size = 2
numworker = 0
epochs = 1000
time_dim = 128
num_classes = 2
Lambda = 100
seed = 42

exp = "exp_1/"
whole_Abeta = "./data/whole_SPECT"
latent_Abeta = "./data/latent_SPECT/"
whole_MRI = "./data/whole_MRI"
path = "/workspace/projects/T1-SPECT-translation/IL-CLDM/data/whole_MRI/3102.nii"
gpus = [0]

CHECKPOINT_AAE = "result/"+exp+"AAE.pth.tar"
CHECKPOINT_Unet = "result/"+exp+"Unet.pth.tar"

# Spatial settings
# Crop size applied to MRI/SPECT before AAE and to MRI in LDM. Latent dims are crop_size/4.
# Choose a cubic size divisible by 8 to avoid odd shapes in UNet down/upsampling.
crop_size = (64, 64, 64)
latent_shape = (crop_size[0] // 4, crop_size[1] // 4, crop_size[2] // 4)

train = "data_info/train.txt"
validation = "data_info/validation.txt"
test = "data_info/test.txt"

# Optional manual resume hints (used if checkpoints lack metadata).
resume_epoch = None  # e.g., 787
resume_average = None  # e.g., best psnr+ssim*10 metric when known
resume_epoch_unet = None
resume_average_unet = None
