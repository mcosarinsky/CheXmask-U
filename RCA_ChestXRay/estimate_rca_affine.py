import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from medpy.metric import dc
from siameseNet_affine import SiameseReg
from unet import UNet
from skimage import transform


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


def apply_registration_mask(mask, params):
    params = params.cpu().numpy()
    affine_matrix = np.zeros((2, 3))
    affine_matrix[0, :] = params[0, 0:3]  # Changed from 0:3 to 0:2
    affine_matrix[1, :] = params[0, 3:6]  # Changed from 3:6 to 2:4
    affine_matrix[:2,2] = affine_matrix[:2,2] * 1024 
    img = cv2.warpAffine(mask.astype('float'), affine_matrix, (mask.shape[1], mask.shape[0]))
    return img

def load_models(config):
    modelReg = SiameseReg(config).float().to(config['device'])
    modelReg.load_state_dict(torch.load(
        "/home/ngaggion/DATA/X-Ray/RigidReg/Training/affine_pretrained_with_kl/bestMSE.pt"), strict=False)
    modelReg.eval()

    modelFinder = SiameseReg(config).float().to(config['device'])
    modelFinder.load_state_dict(torch.load("/home/ngaggion/DATA/X-Ray/RigidReg/bestMSE.pt"), strict=False)
    modelFinder.eval()

    modelPreds = []
    for i in range(6):
        modelPred = UNet(n_classes=2).to(config['device'])
        modelPred.load_state_dict(torch.load(f"/home/ngaggion/DATA/X-Ray/RigidReg/Training/unet_for_rca/epoch{i}.pt"), strict=False)
        modelPred.eval()
        modelPreds.append(modelPred)

    return modelReg, modelFinder, modelPreds


def process_images(images_test, images_train, latent_matrix, model_finder, model_reg, model_preds, config):
    # Initialize a DataFrame to store the results
    df_reg = pd.DataFrame(columns=['Dice_Real', 'Dice_RCA_Max', 'Dice_RCA_Mean'])

    with torch.no_grad():
        for image in images_test:
            image_path = "../Chest-xray-landmark-dataset/Images/" + image
            img = cv2.imread(image_path, 0) / 255.0
            source = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(config['device']).float()
            RL, LL = load_landmarks(image)
            mask_GT = landmark_to_mask(RL, LL)

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

                real_dice = dc(pred, mask_GT)

                rca_dice_list = []

                for j in range(0, len(gt_params)):
                    params = gt_params[j]
                    mask = gt_masks[j]

                    # Apply registration on the predicted mask using the calculated parameters
                    pred_reg = apply_registration_mask(pred, params)
                    
                    mean = pred_reg.mean()
                    pred_reg = pred_reg > mean

                    rca_dice = dc(pred_reg, mask)
                    rca_dice_list.append(rca_dice)

                rca_max = max(rca_dice_list)
                rca_avg = np.mean(rca_dice_list)

                df_reg.loc[len(df_reg)] = [real_dice, rca_max, rca_avg]

                print("Image:", image, "Real Dice:", real_dice, "RCA Max:", rca_max, "RCA Avg:", rca_avg)

    return df_reg


def calculate_ground_truth(image_names, config, modelReg, source):
    gt_params = []
    gt_masks = []

    for img_near in image_names:
        img_target = cv2.imread("../Chest-xray-landmark-dataset/Images/" + img_near, 0) / 255.0
        target = torch.from_numpy(img_target).unsqueeze(0).unsqueeze(0).to(config['device']).float()
        
        RL_, LL_ = load_landmarks(img_near)
        mask_gt = landmark_to_mask(RL_, LL_)

        params = modelReg(target, source).detach()

        gt_params = gt_params + [params]
        gt_masks = gt_masks + [mask_gt]

    return gt_params, gt_masks


if __name__ == '__main__':
    images_train = open("train_images_lungs.txt", 'r').read().splitlines()

    config = {
        'latents': 64,
        'inputsize': 1024,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'sampling': False
    }

    modelReg, modelFinder, modelPreds = load_models(config)

    latent_space = np.load("latent_space_train.npy")
    latent_matrix = torch.from_numpy(latent_space).to(config['device'])

    images_test = open("test_images_lungs.txt", 'r').read().splitlines()

    # Load the models
    model_reg, model_finder, model_preds = load_models(config)

    # Process the images and generate the DataFrame with the results
    df_rca = process_images(images_test, images_train, latent_matrix, model_finder, model_reg, model_preds, config)

    # Save the DataFrame to a CSV file
    df_rca.to_csv("df_rca_affine.csv")
