import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

def merge_mimic():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/MIMIC/MIMIC_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/MIMIC'
    
    df = pd.read_csv(images_path, usecols=['Image'])
    df['dicom_id'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    h, w = 1024, 1024
    df['Height'] = h
    df['Width'] = w

    # Prepare lists to store computed values
    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for img_path in tqdm(df['Image'], desc="Processing images"):
        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/MIMIC/Images/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)
        landmark_mean_list.append(mean)
        landmark_std_list.append(std)

        mean_vals = np.array([int(v) for v in mean.split(",")]).reshape(-1, 2)
        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    # Assign all at once
    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_padchest():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/Padchest/Padchest_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/Padchest'

    df = pd.read_csv(images_path, usecols=['Image'])
    df['ImageID'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]) + '.png'
    h, w = 1024, 1024
    df['Height'] = h
    df['Width'] = w

    # Prepare lists to store computed values
    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for img_path in df['Image']:
        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/Padchest/Images/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)
        landmark_mean_list.append(mean)
        landmark_std_list.append(std)

        mean_vals = np.array([int(v) for v in mean.split(",")]).reshape(-1, 2)
        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    # Assign all at once
    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_padchest_orig():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/Padchest/Padchest_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/Padchest'
    hw_path = '/media/ngaggion/DATA/chexmask-database/OriginalResolution/Padchest.csv'

    df = pd.read_csv(images_path, usecols=['Image'])
    df_hw = pd.read_csv(hw_path, usecols=['ImageID', 'Height', 'Width'])
    df['ImageID'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]) + '.png'
    df = df.merge(df_hw, on='ImageID', how='inner')

    if df.isna().any().any():
        raise ValueError("There are missing values in the merged DataFrame. Please check the ImageID matching.")
        return 

    # Prepare lists to store computed values
    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for idx, row in df.iterrows():
        img_path = row['Image']
        h = row['Height']
        w = row['Width']

        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/Padchest/Images/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)

        # Remap to original resolution
        mean_vals, mean_str_out, std_str_out = remap_landmarks_and_std(mean, std, h, w)

        landmark_mean_list.append(mean_str_out)
        landmark_std_list.append(std_str_out)

        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask  = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    # Assign all at once
    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_mimic_orig():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/MIMIC/MIMIC_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/MIMIC'
    hw_path = '/media/ngaggion/DATA/chexmask-database/OriginalResolution/MIMIC-CXR-JPG.csv'

    df = pd.read_csv(images_path, usecols=['Image'])
    df_hw = pd.read_csv(hw_path, usecols=['dicom_id', 'Height', 'Width'])
    df['dicom_id'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df = df.merge(df_hw, on='dicom_id', how='inner')

    if df.isna().any().any():
        raise ValueError("There are missing values in the merged DataFrame. Please check the dicom_id matching.")
        return 

    # Prepare lists to store computed values
    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for idx, row in df.iterrows():
        img_path = row['Image']
        h = row['Height']
        w = row['Width']

        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/MIMIC/Images/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)

        # Remap to original resolution
        mean_vals, mean_str_out, std_str_out = remap_landmarks_and_std(mean, std, h, w)

        landmark_mean_list.append(mean_str_out)
        landmark_std_list.append(std_str_out)

        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask  = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    # Assign all at once
    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_vindr():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/VinBigData/VinBigData_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/VinDr'

    df = pd.read_csv(images_path, usecols=['Image'])  # change column name if different
    df['image_id'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    h, w = 1024, 1024
    df['Height'] = h
    df['Width'] = w

    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for img_path in df['Image']:
        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/VinBigData/pngs/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)
        landmark_mean_list.append(mean)
        landmark_std_list.append(std)

        mean_vals = np.array([int(v) for v in mean.split(",")]).reshape(-1, 2)
        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_vindr_orig():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/VinBigData/VinBigData_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/VinDr'
    hw_path = '/media/ngaggion/DATA/chexmask-database/OriginalResolution/VinDr-CXR.csv'

    df = pd.read_csv(images_path, usecols=['Image'])  # adjust if column name differs
    df_hw = pd.read_csv(hw_path, usecols=['ImageID', 'Height', 'Width'])  # adjust if needed
    df['image_id'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df = df.merge(df_hw, on='ImageID', how='inner')

    if df.isna().any().any():
        raise ValueError("There are missing values in the merged DataFrame. Please check the ImageID matching.")
        return 

    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for idx, row in df.iterrows():
        img_path = row['Image']
        h = row['Height']
        w = row['Width']

        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/VinBigData/pngs/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)

        mean_vals, mean_str_out, std_str_out = remap_landmarks_and_std(mean, std, h, w)

        landmark_mean_list.append(mean_str_out)
        landmark_std_list.append(std_str_out)

        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask  = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df



def merge_chexpert():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/CheXpert/CHEXPERT_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/CheXpert'

    df = pd.read_csv(images_path, usecols=['Image'])  # change column name if different
    def last_3_parents_plus_stem(p):
        p = Path(p)
        parts = list(p.parts)
        return Path(*parts[-4:-1], p.stem).as_posix()

    df["Path"] = df["Image"].apply(last_3_parents_plus_stem)
    h, w = 1024, 1024
    df['Height'] = h
    df['Width'] = w

    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for img_path in df['Image']:
        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/CheXpert/Preprocessed/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)
        landmark_mean_list.append(mean)
        landmark_std_list.append(std)

        mean_vals = np.array([int(v) for v in mean.split(",")]).reshape(-1, 2)
        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_chexpert_orig():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/CheXpert/CHEXPERT_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/CheXpert'
    hw_path = '/media/ngaggion/DATA/chexmask-database/OriginalResolution/CheXpert.csv'

    df_hw = pd.read_csv(hw_path, usecols=['Path', 'Height', 'Width'])  # adjust if needed
    df = pd.read_csv(images_path, usecols=['Image'])  # change column name if different
    def last_3_parents_plus_stem(p):
        p = Path(p)
        parts = list(p.parts)
        return Path(*parts[-4:-1], p.stem).as_posix()

    
    df["Path"] = df["Image"].apply(last_3_parents_plus_stem) + '.jpg'
    df = df.merge(df_hw, on='Path', how='inner')

    if df.isna().any().any():
        raise ValueError("There are missing values in the merged DataFrame. Please check the ImageID matching.")
        return 

    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for idx, row in df.iterrows():
        img_path = row['Image']
        h = row['Height']
        w = row['Width']

        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/CheXpert/Preprocessed/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)

        mean_vals, mean_str_out, std_str_out = remap_landmarks_and_std(mean, std, h, w)

        landmark_mean_list.append(mean_str_out)
        landmark_std_list.append(std_str_out)

        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask  = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_chestx():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/Chest8/CHEST8_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/ChestX-ray8'

    df = pd.read_csv(images_path, usecols=['Image'])  # change column name if different
    df['Image Index'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]) + '.png'

    h, w = 1024, 1024
    df['Height'] = h
    df['Width'] = w

    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for img_path in df['Image']:
        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/Chest8/ChestX-ray8/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)
        landmark_mean_list.append(mean)
        landmark_std_list.append(std)

        mean_vals = np.array([int(v) for v in mean.split(",")]).reshape(-1, 2)
        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def merge_vindr_orig():
    images_path = '/media/ngaggion/DATA/X-Ray/BigScale/Datasets/VinBigData/VinBigData_RCA.csv'
    outputs_path = '/media/ngaggion/DATA/HybridGNet-uncertainty/Outputs/Predictions/CheXmask/VinDr'
    hw_path = '/media/ngaggion/DATA/chexmask-database/OriginalResolution/VinDr-CXR.csv'

    df = pd.read_csv(images_path, usecols=['Image'])  # adjust if column name differs
    df_hw = pd.read_csv(hw_path, usecols=['image_id', 'Height', 'Width'])  # adjust if needed
    df['image_id'] = df['Image'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df = df.merge(df_hw, on='image_id', how='inner')

    if df.isna().any().any():
        raise ValueError("There are missing values in the merged DataFrame. Please check the ImageID matching.")
        return 

    landmark_mean_list = []
    landmark_std_list = []
    left_lung_list = []
    right_lung_list = []
    heart_list = []

    for idx, row in df.iterrows():
        img_path = row['Image']
        h = row['Height']
        w = row['Width']

        base = img_path.replace('/home/ngaggion/DATA/X-Ray/BigScale/Datasets/VinBigData/pngs/', '')
        base = os.path.splitext(base)[0]
        csv_path = os.path.join(outputs_path, f'{base}.csv')

        if not os.path.exists(csv_path):
            print(f'File not found: {csv_path}')
            return

        res = pd.read_csv(csv_path)
        mean, std = extract_mean_std_strings(res)

        mean_vals, mean_str_out, std_str_out = remap_landmarks_and_std(mean, std, h, w)

        landmark_mean_list.append(mean_str_out)
        landmark_std_list.append(std_str_out)

        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        RL_mask = get_RLE_from_mask(getDenseMask(RL, h, w))
        LL_mask = get_RLE_from_mask(getDenseMask(LL, h, w))
        H_mask  = get_RLE_from_mask(getDenseMask(H, h, w))

        right_lung_list.append(RL_mask)
        left_lung_list.append(LL_mask)
        heart_list.append(H_mask)

    df['Landmarks (Mean)'] = landmark_mean_list
    df['Landmarks (Std)'] = landmark_std_list
    df['Left Lung'] = left_lung_list
    df['Right Lung'] = right_lung_list
    df['Heart'] = heart_list

    return df


def remap_landmarks_and_std(mean_str, std_str, h, w, preproc_size=1024):
    """
    Map Mean + Std landmarks from preprocessed 1024x1024 padded space
    back to original resolution (h, w).
    """
    scale = preproc_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_left = (preproc_size - new_w) // 2
    pad_top = (preproc_size - new_h) // 2

    mean_vals = np.array([float(v) for v in mean_str.split(",")]).reshape(-1, 2)
    mean_vals[:, 0] -= pad_left
    mean_vals[:, 1] -= pad_top
    mean_vals[:, 0] /= scale
    mean_vals[:, 1] /= scale
    mean_vals = np.clip(np.round(mean_vals), 0, [w - 1, h - 1]).astype(int)

    if std_str is not None and isinstance(std_str, str) and len(std_str) > 0:
        std_vals = np.array([float(v) for v in std_str.split(",")]).reshape(-1, 2)
        std_vals[:, 0] /= scale
        std_vals[:, 1] /= scale
        std_str_out = ",".join([f"{sx:.2f},{sy:.2f}" for sx, sy in std_vals])
    else:
        std_str_out = None

    mean_str_out = ",".join([f"{x},{y}" for x, y in mean_vals])
    return mean_vals, mean_str_out, std_str_out


def extract_mean_std_strings(df):
    # Mean: round and convert to int, concatenate as x1,y1,x2,y2,...
    mean_str = ",".join([f"{int(round(x))},{int(round(y))}" 
                         for x, y in zip(df["Mean x"], df["Mean y"])])

    # Std: round to 2 decimals, concatenate as x1.xx,y1.yy,...
    std_str = ",".join([f"{x:.2f},{y:.2f}" 
                        for x, y in zip(df["Std x"], df["Std y"])])

    return mean_str, std_str


def get_RLE_from_mask(mask: np.ndarray) -> str:
    mask = (mask > 0).astype(np.uint8)
    pixels = mask.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def getDenseMask(graph: np.ndarray, h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    graph = graph.reshape(-1, 1, 2).astype(int)
    cv2.drawContours(img, [graph], -1, 255, -1)
    return img

mimic_df = merge_mimic_orig()
mimic_df.to_csv('/media/ngaggion/DATA/chexmask-database/OriginalResolution/MIMIC-CXR-JPG.csv', index=False)

padchest_df = merge_padchest_orig()
padchest_df.to_csv('/media/ngaggion/DATA/chexmask-database/OriginalResolution/Padchest.csv', index=False)

vindr_df = merge_vindr()
vindr_df.to_csv('/media/ngaggion/DATA/chexmask-database/Preprocessed/VinDr-CXR.csv', index=False)

df_chest8 = merge_chestx()
df_chest8.to_csv('/media/ngaggion/DATA/chexmask-database/OriginalResolution/ChestX-ray8.csv', index=False)

#print('Merging preprocessed')
df_chexpert = merge_chexpert()
df_chexpert.to_csv('/media/ngaggion/DATA/chexmask-database/Preprocessed/CheXpert.csv', index=False)

print('Merging orig')
df_chexpert = merge_chexpert_orig()
df_chexpert.to_csv('/media/ngaggion/DATA/chexmask-database/OriginalResolution/CheXpert_2.csv', index=False)

vindr_df = merge_vindr_orig()
vindr_df.to_csv('/media/ngaggion/DATA/chexmask-database/OriginalResolution/VinDr-CXR.csv', index=False)