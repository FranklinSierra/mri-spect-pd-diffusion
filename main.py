import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from utils import *
from model import *
from dataset import *
import copy
import config
import csv
import os
import nibabel as nib

import warnings
warnings.filterwarnings("ignore")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=0.0001, beta_end=0.0200):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(config.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, y):
        with torch.no_grad():
            n = y.shape[0]
            x = torch.randn((n, 1, *config.latent_shape)).to(config.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(config.device)
                predicted_noise = model(x, y, t)
                alpha = self.alpha[t][:, None, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None, None]
                beta = self.beta[t][:, None, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x

# First stage
def train_AAE():
    # Ensure result directory exists to avoid FileNotFoundError on fresh runs.
    os.makedirs(os.path.join("result", str(config.exp)), exist_ok=True)
    seed_torch(config.seed)
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.999)) 
    disc = Discriminator().to(config.device)
    opt_disc = optim.Adam(disc.parameters(),lr=config.learning_rate,betas=(0.5, 0.999))

    start_epoch = 0
    average = 0
    patience = 0
    # Resume if checkpoint exists.
    if os.path.exists(config.CHECKPOINT_AAE):
        ckpt = load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
        # If old checkpoints lack metadata, infer epoch/best metric from logs.
        start_epoch = ckpt.get("epoch")
        if start_epoch is None:
            loss_log = os.path.join("result", str(config.exp), "loss_curve.csv")
            if os.path.exists(loss_log):
                try:
                    with open(loss_log, newline="") as f:
                        reader = csv.reader(f)
                        next(reader, None)  # skip header
                        for row in reader:
                            if row and row[0].strip().isdigit():
                                start_epoch = int(row[0])
                except Exception:
                    start_epoch = 0
            if start_epoch is None:
                start_epoch = 0

        average = ckpt.get("average")
        if average is None:
            val_log = os.path.join("result", str(config.exp), "validation.csv")
            if os.path.exists(val_log):
                try:
                    with open(val_log, newline="") as f:
                        reader = csv.reader(f)
                        next(reader, None)
                        average = 0
                        for row in reader:
                            if len(row) >= 3:
                                try:
                                    psnr_val = float(row[1])
                                    ssim_val = float(row[2])
                                    average = max(average, psnr_val + ssim_val * 10)
                                except ValueError:
                                    continue
                except Exception:
                    average = 0
        if average is None:
            average = 0
        patience = ckpt.get("patience", 0)

    # Allow manual override if checkpoint lacks metadata and logs were wiped.
    if config.resume_epoch is not None:
        start_epoch = config.resume_epoch
    if config.resume_average is not None:
        average = config.resume_average

        print(f"=> Resuming AAE from epoch {start_epoch}, best metric {average}")

    for epoch in range(start_epoch, config.epochs):
        print("epoch:", epoch)
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","recon_loss","disc_loss_epoch"])
        
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task = config.train, stage = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        recon_loss_epoch=0
        disc_loss_epoch=0

        for idx, (Abeta, stage) in enumerate(loop):
            #print("Min and max of Abeta batch:", Abeta.min().item(), Abeta.max().item())
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            decoded_Abeta = model(Abeta)
            
            disc_real = disc(Abeta)
            disc_fake = disc(decoded_Abeta)

            recon_loss = torch.abs(Abeta - decoded_Abeta).mean()
            g_loss = -torch.mean(disc_fake)
            loss = recon_loss*config.Lambda + g_loss

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            disc_loss = (d_loss_real + d_loss_fake)/2

            opt_model.zero_grad()
            loss.backward(retain_graph=True)

            opt_disc.zero_grad()
            disc_loss.backward()

            opt_model.step()
            opt_disc.step()

            recon_loss_epoch = recon_loss_epoch + recon_loss
            disc_loss_epoch = disc_loss_epoch + disc_loss
        
        writer.writerow([epoch+1, recon_loss_epoch.item()/length, disc_loss_epoch.item()/length])
        lossfile.close()

        #validation part
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.validation, stage= "validation")
        loader = DataLoader(dataset,batch_size= 1,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        psnr_0 = 0
        ssim_0 = 0

        csvfile = open("result/"+str(config.exp)+"validation.csv", 'a+',newline = '')
        writer = csv.writer(csvfile)
        if epoch == 0:
            writer.writerow(['Epoch','PSNR','SSIM'])

        for idx, (Abeta, stage) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            
            decoded_Abeta = model(Abeta)
            decoded_Abeta = torch.clamp(decoded_Abeta,0,1)
            decoded_Abeta = decoded_Abeta.detach().cpu().numpy()
            decoded_Abeta = np.squeeze(decoded_Abeta)
            decoded_Abeta = decoded_Abeta.astype(np.float32)

            Abeta = Abeta.detach().cpu().numpy()
            Abeta = np.squeeze(Abeta)
            Abeta = Abeta.astype(np.float32)
            data_range = max(Abeta.max() - Abeta.min(), 1e-8)
        
            psnr_0 += round(psnr(Abeta,decoded_Abeta, data_range=data_range),3)
            ssim_0 += round(ssim(Abeta, decoded_Abeta, data_range=data_range), 3)
        
        average_epoch = psnr_0/length + ssim_0 * 10/length
        writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
        csvfile.close()
        
        # test part
        if average_epoch > average:
            average = average_epoch
            patience = 0
            # Save epoch/average to make resuming possible.
            save_checkpoint(model, opt_model, filestage=config.CHECKPOINT_AAE, epoch=epoch+1, average=average, patience=patience)

            dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.test, stage= "test")
            loader = DataLoader(dataset,batch_size= 1,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.length_dataset
            psnr_0 = 0
            ssim_0 = 0
            
            csvfile = open("result/"+str(config.exp)+"test.csv", 'a+',newline = '')
            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(['Epoch','PSNR','SSIM'])

            for idx, (Abeta, stage) in enumerate(loop):
                Abeta = np.expand_dims(Abeta, axis=1)
                Abeta = torch.tensor(Abeta)
                Abeta = Abeta.to(config.device)
                
                decoded_Abeta = model(Abeta)
                decoded_Abeta = torch.clamp(decoded_Abeta,0,1)
                decoded_Abeta = decoded_Abeta.detach().cpu().numpy()
                decoded_Abeta = np.squeeze(decoded_Abeta)
                decoded_Abeta = decoded_Abeta.astype(np.float32)

                Abeta = Abeta.detach().cpu().numpy()
                Abeta = np.squeeze(Abeta)
                Abeta = Abeta.astype(np.float32)
                data_range = max(Abeta.max() - Abeta.min(), 1e-8)
            
                psnr_0 += round(psnr(Abeta,decoded_Abeta, data_range=data_range),3)
                ssim_0 += round(ssim(Abeta, decoded_Abeta, data_range=data_range), 3)

            writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
            csvfile.close()
        else:
            patience += 1
            if patience >= 15:
                print(f"Early stopping AAE (patience=15) at epoch {epoch+1}, best metric {average}")
                break

def encoding():
    seed_torch(config.seed)
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    print("checkpoint loaded! from: ", config.CHECKPOINT_AAE)
    image = nib.load(config.path)
    #print("image loaded!", image.shape)

    dataset = OneDataset(root_Abeta=config.whole_Abeta, task = config.train, stage= "Non")
    #print("OneDataset created!")
    loader = DataLoader(dataset,batch_size=1,shuffle=True,num_workers=config.numworker,pin_memory=True)
    #print("DataLoader created!")
    loop = tqdm(loader, leave=True)
    #print("TQDM loop created!", loop)

    for idx, (Abeta,stage) in enumerate(loop):
        #print("idx:", idx)
        #print("Abeta shape:", Abeta.shape)
        #print("Stage:", stage)
        Abeta = np.expand_dims(Abeta, axis=1)
        Abeta = torch.tensor(Abeta)
        Abeta = Abeta.to(config.device)
        
        latent_Abeta = model.encoder(Abeta)
        latent_Abeta = latent_Abeta.detach().cpu().numpy()
        latent_Abeta = np.squeeze(latent_Abeta)
        latent_Abeta = latent_Abeta.astype(np.float32)

        latent_Abeta = nib.Nifti1Image(latent_Abeta, image.affine)
        nib.save(latent_Abeta, config.latent_Abeta+str(stage[0]))

# Second stage
def train_LDM():
    seed_torch(config.seed)
    gpus = config.gpus
    os.makedirs(os.path.join("result", str(config.exp)), exist_ok=True)
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(),lr=config.learning_rate,betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    print("AAE checkpoint loaded! from: ", config.CHECKPOINT_AAE)

    Unet = UNet(in_channel=2, out_channel=1, image_size=config.latent_shape[0]).to(config.device)
    opt_Unet= optim.AdamW(Unet.parameters(), lr=config.learning_rate)
    Unet = nn.DataParallel(Unet,device_ids=gpus,output_device=gpus[0])
    ema = EMA(0.9999)
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)

    L2 = nn.MSELoss()
    diffusion = Diffusion()
    # Latent stats for normalization
    latent_mean, latent_std = compute_latent_stats(config.latent_Abeta)
    latent_mean_t = torch.tensor(latent_mean, device=config.device, dtype=torch.float32).view(1,1,1,1,1)
    latent_std_t = torch.tensor(latent_std, device=config.device, dtype=torch.float32).view(1,1,1,1,1)
    print(f"Latent stats -> mean: {latent_mean:.4f}, std: {latent_std:.4f}")
    start_epoch = 0
    average = 0
    best_ssim = -1e9
    patience = 0
    # Resume if checkpoint exists.
    if os.path.exists(config.CHECKPOINT_Unet):
        ckpt = load_checkpoint(config.CHECKPOINT_Unet, ema_Unet, opt_Unet, config.learning_rate)
        # Load Unet weights if stored separately; otherwise fall back to EMA weights.
        if "unet_state_dict" in ckpt:
            Unet.load_state_dict(ckpt["unet_state_dict"])
        else:
            Unet.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        average = ckpt.get("average", 0)
        best_ssim = ckpt.get("best_ssim", best_ssim)
        patience = ckpt.get("patience", patience)
    # Allow manual overrides if checkpoint lacks metadata.
    if config.resume_epoch_unet is not None:
        start_epoch = config.resume_epoch_unet
    if config.resume_average_unet is not None:
        average = config.resume_average_unet
    if start_epoch or average:
        print(f"=> Resuming Unet from epoch {start_epoch}, best metric {average}")

    def safe_ssim(gt, pred, data_range):
        # Adapt window size to the smallest spatial dim; if too small, return 0.
        min_side = min(gt.shape)
        win_size = 7
        if min_side < win_size:
            win_size = min_side if (min_side % 2 == 1) else max(min_side - 1, 1)
        if win_size < 3 or min_side < 2:
            return 0.0
        return ssim(gt, pred, data_range=data_range, win_size=win_size)

    for epoch in range(start_epoch, config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","MSE_loss"])
        
        dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.latent_Abeta, task = config.train, stage = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset

        MSE_loss_epoch = 0

        for idx, (MRI, latent_Abeta, stage, label) in enumerate(loop):
            #print("min and max of mri batch:", MRI.min().item(), MRI.max().item())
            #print("min and max of latent_Abeta batch:", latent_Abeta.min().item(), latent_Abeta.max().item())
            label = label.to(config.device)
            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI, device=config.device)
            latent_Abeta = np.expand_dims(latent_Abeta, axis=1)
            latent_Abeta = torch.tensor(latent_Abeta, device=config.device)
            latent_Abeta = (latent_Abeta - latent_mean_t) / latent_std_t

            t = diffusion.sample_timesteps(latent_Abeta.shape[0]).to(config.device)
            x_t, noise = diffusion.noise_images(latent_Abeta, t)
            predicted_noise = Unet(x_t, MRI, t, label)
            loss = L2(predicted_noise, noise)

            opt_Unet.zero_grad()
            loss.backward()
            opt_Unet.step()
            ema.step_ema(ema_Unet, Unet)

            MSE_loss_epoch += loss

        writer.writerow([epoch+1,MSE_loss_epoch.item()/length])
        lossfile.close()

        #validation part
        dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.whole_Abeta, task = config.validation, stage = "validation")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
        loop = tqdm(loader, leave=True)
        length = dataset.length_dataset
        psnr_0 = 0
        ssim_0 = 0

        csvfile = open("result/"+str(config.exp)+"validation.csv", 'a+',newline = '')
        writer = csv.writer(csvfile)
        if epoch == 0:
            writer.writerow(['Epoch','PSNR','SSIM'])

        for idx, (MRI, Abeta, stage, label) in enumerate(loop):
            MRI = np.expand_dims(MRI, axis=1)
            MRI = torch.tensor(MRI, device=config.device)

            sampled_latent = diffusion.sample(ema_Unet, MRI)
            sampled_latent = sampled_latent * latent_std_t + latent_mean_t
            syn_Abeta = model.decoder(sampled_latent)
            syn_Abeta = torch.clamp(syn_Abeta,0,1)
            syn_Abeta = syn_Abeta.detach().cpu().numpy()
            syn_Abeta = np.squeeze(syn_Abeta)
            syn_Abeta = syn_Abeta.astype(np.float32)

            # DataLoader convierte los numpy en torch; pasa a numpy antes de métricas.
            Abeta = Abeta.detach().cpu().numpy()
            Abeta = np.squeeze(Abeta)
            Abeta = Abeta.astype(np.float32)
            data_range = max(Abeta.max() - Abeta.min(), 1e-8)

            psnr_0 += round(psnr(Abeta,syn_Abeta, data_range=data_range),3)
            ssim_0 += round(safe_ssim(Abeta, syn_Abeta, data_range=data_range), 3)
        
        psnr_avg = psnr_0/length
        ssim_avg = ssim_0/length
        average_epoch = psnr_avg + ssim_avg * 10
        writer.writerow([epoch+1, psnr_avg, ssim_avg])
        csvfile.close()
        # Early stopping on validation SSIM with patience.
        if ssim_avg > best_ssim:
            best_ssim = ssim_avg
            patience = 0
        else:
            patience += 1
        if patience >= 15:
            print(f"Early stopping (patience=15) at epoch {epoch+1}, best SSIM {best_ssim}")
            break
        
        # test part
        if average_epoch > average:
            average = average_epoch
            save_checkpoint(
                ema_Unet,
                opt_Unet,
                filestage=config.CHECKPOINT_Unet,
                epoch=epoch+1,
                average=average,
                unet_state_dict=Unet.state_dict(),
                best_ssim=best_ssim,
                patience=patience,
            )

            dataset = TwoDataset(root_MRI=config.whole_MRI, root_Abeta=config.whole_Abeta, task = config.test, stage = "test")
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True)
            loop = tqdm(loader, leave=True)
            length = dataset.length_dataset
            psnr_0 = 0
            ssim_0 = 0

            csvfile = open("result/"+str(config.exp)+"test.csv", 'a+',newline = '')
            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(['Epoch','PSNR','SSIM'])

            for idx, (MRI, Abeta, stage, label) in enumerate(loop):
                MRI = np.expand_dims(MRI, axis=1)
                MRI = torch.tensor(MRI, device=config.device)

                sampled_latent = diffusion.sample(ema_Unet, MRI)
                sampled_latent = sampled_latent * latent_std_t + latent_mean_t
                syn_Abeta = model.decoder(sampled_latent)
                syn_Abeta = torch.clamp(syn_Abeta,0,1)
                syn_Abeta = syn_Abeta.detach().cpu().numpy()
                syn_Abeta = np.squeeze(syn_Abeta)
                syn_Abeta = syn_Abeta.astype(np.float32)

                Abeta = Abeta.detach().cpu().numpy()
                Abeta = np.squeeze(Abeta)
                Abeta = Abeta.astype(np.float32)
                data_range = max(Abeta.max() - Abeta.min(), 1e-8)

                psnr_0 += round(psnr(Abeta,syn_Abeta, data_range=data_range),3)
                ssim_0 += round(safe_ssim(Abeta, syn_Abeta, data_range=data_range), 3)

            writer.writerow([epoch+1, psnr_0/length, ssim_0/length])
            csvfile.close()

if __name__ == '__main__':
    print("device:", config.device)
    seed_torch(config.seed)
    train_AAE()
    encoding()
    train_LDM()
