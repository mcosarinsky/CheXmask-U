import os
import torch
from siameseNet import SiameseReg
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
modelReg1.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/Training/frozen_til_100_then_unfroze/bestMSE.pt"), strict=False)
modelReg1.eval()

modelReg2 = SiameseReg(config).float().to(config['device'])
modelReg2.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/Training/pretrained_finetune_from_0_with_KL/bestMSE.pt"), strict=False)
modelReg2.eval()

modelReg3 = SiameseReg(config).float().to(config['device'])
modelReg3.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/Training/pretrained_finetune_from_0_no_KL/bestMSE.pt"), strict=False)
modelReg3.eval()

models_reg = [modelReg1, modelReg2, modelReg3]
models_reg_names = ['pretrained_frozen', 'pretrained_finetune_from_0_with_KL', 'pretrained_finetune_from_0_no_KL']

modelFinder = SiameseReg(config).float().to(config['device'])
modelFinder.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/bestMSE.pt"), strict=False)
# modelFinder.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/BigScale/Baselines/Training/Prod_LH_90_10/bestMSE.pt"), strict=False)

modelFinder.eval()

print('Loaded')

'''

model = SiameseReg(config).float().to(config['device'])
# model.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/BigScale/Baselines/Training/Prod_LH_90_10/bestMSE.pt"), strict=False)
model.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/bestMSE.pt"), strict=False)

encodings = []

for i in range(len(images_train)):
    print(i)
    image_path = "../Chest-xray-landmark-dataset/Images/" + images_train[i]

    t1 = time.time()
    image = cv2.imread(image_path, 0) / 255.0
    t2 = time.time()
    data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(config['device']).float()
    t3 = time.time()
    mu, _ = model.encoder(data)
    t4 = time.time()
    encodings.append(mu.detach().cpu().numpy())
    t5 = time.time()

    print("Times. Read", t2-t1, "Torch", t3-t2, "Encoder", t4-t3, "Append", t5-t4)

latent_space = np.concatenate(encodings, axis=0)
np.save("latent_space_train.npy", latent_space)
'''

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

def applyRegistrationLandmarks(RL, LL, params):
    alfa = params[0,0].cpu().numpy()
    scale = params[0,1:3].cpu().numpy()
    translate = params[0,3:5].cpu().numpy()
    
    landmarks2 = np.concatenate((RL, LL), axis=0)
    
    y2 = torch.tensor(landmarks2).float().to(config['device']) / 1024
    alfa = torch.tensor(alfa).float().to(config['device'])
    rotation_matrix = torch.tensor([[torch.cos(alfa), -torch.sin(alfa)], [torch.sin(alfa), torch.cos(alfa)]]).to(config['device'])

    y2 = y2 - 0.5
    y2 = torch.matmul(y2, rotation_matrix).cpu().numpy()
    y2 = y2 + 0.5
    y2 = y2 * scale + translate
    y2 = y2 * 1024
    
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

            dice_base = dc(mask_base, mask_gt)
            
            if dice_base > dice_base_:
                dice_base_ = dice_base

            if k in Ks:
                df_sel = df_sel.append({'Image': image, 'Dice_base': dice_base_, 'K': k}, ignore_index=True)
                
            for i in range(0, 3):
                modelReg = models_reg[i]     
                modelReg_name = models_reg_names[i]

                params = modelReg(data_, data)
                
                RL_m, LL_m = applyRegistrationLandmarks(RL, LL, params)
                    
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
df_reg.to_csv("df_reg_rigid.csv")
df_sel.to_csv("df_sel.csv")
