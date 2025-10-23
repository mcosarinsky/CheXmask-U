import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from medpy.metric import dc
from siameseNet import SiameseReg
from unet import UNet
from skimage import transform


def load_landmarks(path):
    RL_path = "../Chest-xray-landmark-dataset/landmarks/RL/" + path.replace("png", "npy")
    LL_path = "../Chest-xray-landmark-dataset/landmarks/LL/" + path.replace("png", "npy")
    H_path = "../Chest-xray-landmark-dataset/landmarks/H/" + path.replace("png", "npy")
    RL = np.load(RL_path)
    LL = np.load(LL_path)
    H = np.load(H_path)
    
    return RL, LL, H


def landmark_to_mask(RL, LL, H):
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')
    H = H.reshape(-1, 1, 2).astype('int')
    mask = np.zeros((1024, 1024))
    mask = cv2.drawContours(mask, [RL], -1, 1, -1)
    mask = cv2.drawContours(mask, [LL], -1, 1, -1)
    mask = cv2.drawContours(mask, [H], -1, 2, -1)
    
    return mask


def apply_registration_mask(mask, params):
    alfa = params[0,0].cpu().numpy()
    scale = params[0,1:3].cpu().numpy()
    translate = params[0,3:5].cpu().numpy()

    mask = mask.astype('float32')
    
    # pad image to avoid cropping
    image2 = np.pad(mask, ((250, 250), (250, 250)), 'constant')
    image2 = transform.rotate(image2, alfa*180/np.pi, resize=False)
    image2 = image2[250:1024+250, 250:1024+250]

    h, w = image2.shape[:2]
    new_h = np.round(h * scale[1]).astype('int')
    new_w = np.round(w * scale[0]).astype('int')

    img = transform.resize(image2, (new_h, new_w))

    translate = translate * 1024
    translate = translate.astype('int')

    # translate image by cropping and padding

    if translate[1] < 0:
        img = img[-translate[1]:, :]
        img = np.pad(img, ((0, -translate[1]), (0, 0)), 'constant')
    else:
        img = np.pad(img, ((translate[1], 0), (0, 0)), 'constant')
        img = img[:1024, :]

    if translate[0] < 0:
        img = img[:, -translate[0]:]
        img = np.pad(img, ((0, 0), (0, -translate[0])), 'constant')
    else:
        img = np.pad(img, ((0, 0), (translate[0], 0)), 'constant')
        img = img[:, :1024]

    img = img[:1024, :1024]
    img = np.pad(img, ((0, 1024 - img.shape[0]), (0, 1024 - img.shape[1])), 'constant')
    
    return img


def load_models(config):
    modelReg = SiameseReg(config).float().to(config['device'])
    modelReg.load_state_dict(torch.load(
        "/home/ngaggion/DATA/X-Ray/RigidReg/Training/pretrained_finetune_from_0_with_KL/bestMSE.pt"), strict=False)
    modelReg.eval()

    modelFinder = SiameseReg(config).float().to(config['device'])
    modelFinder.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/bestMSE.pt"), strict=False)
    modelFinder.eval()

    modelPreds = []
    for i in range(10):
        modelPred = UNet(n_classes=3).to(config['device'])
        modelPred.load_state_dict(torch.load(f"/home/ngaggion/DATA/X-Ray/RigidReg/Training/unet_lh_for_rca/epoch{i}.pt"), strict=False)
        modelPred.eval()
        modelPreds.append(modelPred)

    return modelReg, modelFinder, modelPreds


def process_images(images_test, images_train, latent_matrix, model_finder, model_reg, model_preds, config):
    # Initialize a DataFrame to store the results
    df_reg = pd.DataFrame(columns=['Dice_Real', 'Dice_RCA_Max', 'Dice_RCA_Mean', 
                                   'L_Dice_Real', 'L_Dice_RCA_Max', 'L_Dice_RCA_Mean', 
                                   'H_Dice_Real', 'H_Dice_RCA_Max', 'H_Dice_RCA_Mean'])

    with torch.no_grad():
        for image in images_test:
            image_path = "../Chest-xray-landmark-dataset/Images/" + image
            img = cv2.imread(image_path, 0) / 255.0
            source = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(config['device']).float()
            RL, LL, H = load_landmarks(image)
            mask_GT = landmark_to_mask(RL, LL, H)

            # Calculate the latent vector for the current image using the model_finder
            mu, _ = model_finder.encoder(source)
            distances = latent_matrix @ mu.T
            _, sorted_distances_indices = torch.sort(distances, dim=0, descending=True)

            # Select indices of the top 5 nearest images in the latent space
            idxs = sorted_distances_indices[0:5].squeeze().cpu().numpy()
            target_image_names = [images_train[i] for i in idxs]

            # Calculate ground truth parameters and masks for the nearest images
            gt_params, gt_masks = calculate_ground_truth(target_image_names, config, model_reg, source)

            for t in range(0, len(model_preds)):
                pred = model_preds[t](source)[0].argmax(dim=0).cpu().numpy()

                real_dice_lung = dc(pred == 1, mask_GT == 1)
                real_dice_heart = dc(pred == 2, mask_GT == 2)
                real_dice_avg = (real_dice_lung + real_dice_heart) / 2
                
                rca_dice_lung_list = []
                rca_dice_heart_list = []
                rca_dice_avg_list = []

                for j in range(0, len(gt_params)):
                    params = gt_params[j]
                    mask = gt_masks[j]

                    # Apply registration on the predicted mask using the calculated parameters
                    pred_reg = apply_registration_mask(pred, params)
                    
                    lung_mask = pred_reg == 1
                    heart_mask = pred_reg == 2
                    
                    rca_dice_lung = dc(lung_mask, mask == 1)
                    rca_dice_heart = dc(heart_mask, mask == 2)
                    
                    rca_dice_lung_list.append(rca_dice_lung)
                    rca_dice_heart_list.append(rca_dice_heart)
                    rca_dice_avg_list.append((rca_dice_lung + rca_dice_heart) / 2)

                rca_max = max(rca_dice_avg_list)
                rca_avg = np.mean(rca_dice_avg_list)
                rca_max_lung = max(rca_dice_lung_list)
                rca_avg_lung = np.mean(rca_dice_lung_list)
                rca_max_heart = max(rca_dice_heart_list)
                rca_avg_heart = np.mean(rca_dice_heart_list)

                df_reg.loc[len(df_reg)] = [real_dice_avg, rca_max, rca_avg,
                                           real_dice_lung, rca_max_lung, rca_avg_lung,
                                           real_dice_heart, rca_max_heart, rca_avg_heart]

                #print('Image')
                #print('Dice Real: ', real_dice_avg, 'Dice RCA Max: ', rca_max, 'Dice RCA Avg: ', rca_avg)
                #print('Dice Real Lung: ', real_dice_lung, 'Dice RCA Max Lung: ', rca_max_lung, 'Dice RCA Avg Lung: ', rca_avg_lung)
                #print('Dice Real Heart: ', real_dice_heart, 'Dice RCA Max Heart: ', rca_max_heart, 'Dice RCA Avg Heart: ', rca_avg_heart)
                #print("")

    return df_reg


def calculate_ground_truth(image_names, config, modelReg, source):
    gt_params = []
    gt_masks = []

    for img_near in image_names:
        img_target = cv2.imread("../Chest-xray-landmark-dataset/Images/" + img_near, 0) / 255.0
        target = torch.from_numpy(img_target).unsqueeze(0).unsqueeze(0).to(config['device']).float()
        
        RL_, LL_, H_ = load_landmarks(img_near)
        mask_gt = landmark_to_mask(RL_, LL_, H_)

        params = modelReg(target, source).detach()

        gt_params = gt_params + [params]
        gt_masks = gt_masks + [mask_gt]

    return gt_params, gt_masks


if __name__ == '__main__':
    images_train = open("train_images_lungs.txt", 'r').read().splitlines()
    images_train_heart = open("train_images_heart.txt", 'r').read().splitlines()

    # get indices of images with heart in the training set
    idxs = []
    for i in range(len(images_train)):
        if images_train[i] in images_train_heart:
            idxs.append(i)
    
    idxs = np.array(idxs)
    
    config = {
        'latents': 64,
        'inputsize': 1024,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'sampling': False
    }

    modelReg, modelFinder, modelPreds = load_models(config)

    latent_space = np.load("latent_space_train.npy")
    latent_matrix = torch.from_numpy(latent_space).to(config['device'])
    latent_matrix = latent_matrix[idxs, :]

    images_test = open("test_images_heart.txt", 'r').read().splitlines()

    # Load the models
    model_reg, model_finder, model_preds = load_models(config)

    # Process the images and generate the DataFrame with the results
    df_rca = process_images(images_test, images_train_heart, latent_matrix, model_finder, model_reg, model_preds, config)

    # Save the DataFrame to a CSV file
    df_rca.to_csv("df_rca_rigid_lh.csv")
