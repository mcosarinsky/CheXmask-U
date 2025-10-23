import os
import torch
from siameseNet_affine import SiameseReg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from medpy.metric import dc
import pandas as pd
import time

images_train = open("train_images_lungs.txt",'r').read().splitlines()

config = {}
config['latents'] = 64
config['inputsize'] = 1024
config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config['sampling'] = False

modelReg1 = SiameseReg(config).float().to(config['device'])
modelReg1.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/Training/Affine/affine_pretrained_with_kl/bestMSE.pt"), strict=False)
modelReg1.eval()

modelReg2 = SiameseReg(config).float().to(config['device'])
modelReg2.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/Training/Affine/affine_pretrained_no_kl/bestMSE.pt"), strict=False)
modelReg2.eval()

modelReg3 = SiameseReg(config).float().to(config['device'])
modelReg3.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/Training/Affine/affine_frozen/bestMSE.pt"), strict=False)
modelReg3.eval()

models_reg = [modelReg1, modelReg2, modelReg3]
models_reg_names = ['affine with kl', 'affine no kl', 'affine frozen']

modelFinder = SiameseReg(config).float().to(config['device'])
modelFinder.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/bestMSE.pt"), strict=False)
# modelFinder.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/BigScale/Baselines/Training/Prod_LH_90_10/bestMSE.pt"), strict=False)

modelFinder.eval()

print('Loaded')

def load_landmarks(path):
    RL_path = "../Chest-xray-landmark-dataset/landmarks/RL/" + path.replace("png", "npy")
    LL_path = "../Chest-xray-landmark-dataset/landmarks/LL/" + path.replace("png", "npy")
    RL = np.load(RL_path)
    LL = np.load(LL_path)
    return RL, LL

def landmark_to_mask(RL, LL):
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    mask = np.zeros((1024, 1024))
    mask = cv2.drawContours(mask, [RL], -1, 1, -1)
    mask = cv2.drawContours(mask, [LL], -1, 1, -1)
    return mask


def points_to_homogeneous_coordinates(points):
    B, N, _ = points.shape
    ones = torch.ones((B, N, 1), dtype=points.dtype, device=points.device)
    homogeneous_points = torch.cat((points, ones), dim=-1)
    return homogeneous_points

def applyRegistrationLandmarks(RL, LL, params, config):   
    landmarks2 = np.concatenate((RL, LL), axis=0)
    
    affine_matrix = torch.zeros((params.shape[0], 3, 3), dtype=params.dtype, device=params.device)
    affine_matrix[:, 0, :] = params[:, 0:3]  # Changed from 0:3 to 0:2
    affine_matrix[:, 1, :] = params[:, 3:6]  # Changed from 3:6 to 2:4
    affine_matrix[:, 2, 2] = 1
    
    y2 = torch.tensor(landmarks2.astype('float') / 1024).unsqueeze(0).to(config["device"]).float()
    y2 = points_to_homogeneous_coordinates(y2)
    y2 = torch.bmm(affine_matrix, y2.permute(0, 2, 1)).permute(0, 2, 1)[:, :, 0:2]
        
    y2 = y2[0].cpu().numpy() * 1024
    
    RL_pred = y2[:len(RL)]
    LL_pred = y2[len(RL):]
    
    return RL_pred, LL_pred

df_reg = pd.DataFrame(columns=['Image', 'Dice_reg', 'K', 'Model'])
df_sel = pd.DataFrame(columns=['Image', 'Dice_base', 'K'])

latent_space = np.load("latent_space_train.npy")
latentMatrix = torch.from_numpy(latent_space).to(config['device'])

images_test = open("test_images_lungs.txt",'r').read().splitlines()

image_times = []

with torch.no_grad():
    for image in images_test:    
        # print('\r',contador+1,'of', len(all_files),end='')

        t1 = time.time()

        image_path = "../Chest-xray-landmark-dataset/Images/" + image
        
        img = cv2.imread(image_path, 0) / 255.0
        data = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(config['device']).float()
        RL, LL = load_landmarks(image)
        
        mu, _ = modelFinder.encoder(data)       
        distances = latentMatrix @ mu.T
        sorted_distances, sorted_distances_indices = torch.sort(distances, dim=0, descending=True)
        
        Ks = [1,3,5,7,9]
        idxs = sorted_distances_indices[0:Ks[-1]].squeeze().cpu().numpy()
        
        mask_base = landmark_to_mask(RL, LL)

        dice_reg_ = [0, 0, 0]
        dice_base_ = 0

        image_names = [images_train[i] for i in idxs]
        k = 1

        for img_near in image_names:
            img_ = cv2.imread("../Chest-xray-landmark-dataset/Images/" + img_near, 0) / 255.0
            data_ = torch.from_numpy(img_).unsqueeze(0).unsqueeze(0).to(config['device']).float()
            RL_, LL_ = load_landmarks(img_near)
            mask_gt = landmark_to_mask(RL_, LL_)

            for i in range(0, len(models_reg)):
                modelReg = models_reg[i]     
                modelReg_name = models_reg_names[i]

                params = modelReg(data_, data)
                
                RL_m, LL_m = applyRegistrationLandmarks(RL, LL, params, config)
                    
                mask_reg = landmark_to_mask(RL_m, LL_m)
                dice_reg = dc(mask_reg, mask_gt)
                
                if dice_reg > dice_reg_[i]:
                    dice_reg_[i] = dice_reg

                if k in Ks:
                    df_reg = df_reg.append({'Image': image, 'Dice_reg': dice_reg_[i], 
                                            'K': k, 'Model': modelReg_name}, ignore_index=True)  

            k = k + 1

        t2 = time.time()
        image_times.append(t2-t1)

print("Mean time per image:", np.mean(image_times))

# save both dfs
df_reg.to_csv("df_reg_affine.csv")
