import os
import random
import pathlib
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_image_and_samples(img_name, img_dir, skips=True):
    base_name = os.path.splitext(img_name)[0]
    img = cv2.imread(os.path.join(img_dir, img_name), cv2.IMREAD_GRAYSCALE)

    output_dir = img_dir.replace('Images', f"Predictions/{'Skip' if skips else 'NoSkip'}")
    samples_path = os.path.join(output_dir, base_name + '.csv')

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    samples = pd.read_csv(samples_path, index_col=0)
    return samples, img

def extract_landmarks(file_name: str) -> dict:
    file_name = file_name.split('.')[0]  # Remove file extension
    landmarks_dir = 'Annotations'
    landmarks_dict = {}
    
    for key in ['RL', 'LL', 'H']:
        file_path = os.path.join(landmarks_dir, key, file_name + '.npy')
        if os.path.exists(file_path):
            landmarks_dict[key] = np.load(file_path)
        else:
            landmarks_dict[key] = None  
    return landmarks_dict

def sample_landmarks(landmark_dict, organ, n_samples=1):
    if landmark_dict.get(organ) is None:
        return np.empty((0,2))
    else:
        landmarks = landmark_dict[organ]
        n_samples = min(n_samples, len(landmarks))
        
        indices = np.random.choice(len(landmarks), n_samples, replace=False)

        return landmarks[indices]

def get_error(output_dir, file_name):
    base_name = os.path.splitext(file_name)[0]
    samples_path = os.path.join(output_dir, file_name)
    df = pd.read_csv(samples_path, index_col=0)

    landmarks_dict = extract_landmarks(file_name)
    gt = np.concatenate([landmarks_dict[k] for k in ['RL', 'LL', 'H'] if landmarks_dict[k] is not None])
    n_nodes = gt.shape[0]

    pred = np.array(df[['Mean x', 'Mean y']])[:n_nodes]
    sigmas = np.array((df['Std x'] + df['Std y']) / 2)[:n_nodes]

    error = np.linalg.norm(pred - gt, axis=1)

    return error, sigmas

def compute_global_vmax(df_original, df_corrupted):
    """Compute the maximum sigma_avg across both original and corrupted DataFrames."""
    def calc_sigma_avg(df):
        return ((df['Std x'] + df['Std y']) / 2).max()

    max_original = calc_sigma_avg(df_original)
    max_corrupted = calc_sigma_avg(df_corrupted)

    return max(max_original, max_corrupted)


def read_sigma_files(file_path):
    sigma_dict = defaultdict(lambda: {"sigmas": [], "corr_levels": []})

    for folder in natsorted(os.listdir(file_path)):
        subdir = os.path.join(file_path, folder)
        sigma_file = [f for f in os.listdir(subdir) if f.endswith('.txt')][0]

        with open(os.path.join(subdir, sigma_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_name = parts[0].split('.')[0]

                # Latents are stored as variance, take sqrt to get std
                sigmas = np.sqrt(np.array(parts[1:], dtype=np.float32))
                sigma_dict[img_name]['corr_levels'].append(folder)
                sigma_dict[img_name]['sigmas'].append(sigmas)

    return sigma_dict

def process_and_store_sigma(corr_dir):
    sigma_dict = defaultdict(lambda: {"sigmas": [], "corr_levels": []})

    for folder in natsorted(os.listdir(corr_dir)):
        for file in os.listdir(os.path.join(corr_dir, folder)):
            if file.endswith('.csv'):
                sigma = find_avg_std(os.path.join(corr_dir, folder, file))
                file_name = os.path.splitext(file)[0]
                sigma_dict[file_name]["sigmas"].append(sigma)
                sigma_dict[file_name]["corr_levels"].append(folder)
    return sigma_dict


def find_avg_std(file: str):
    df = pd.read_csv(file, index_col=0)
    sigma = (df['Std x'] + df['Std y']) / 2
    return np.array(sigma)


def split_distributions(csv_path, threshold=0.7):
    """
    Splits the DataFrame into in-distribution and out-of-distribution based on a threshold.
    """
    csv_path = pathlib.Path(csv_path).resolve()
    base_dir = csv_path.parent

    df = pd.read_csv(csv_path)
    df['Image'] = df['Image'].apply(
        lambda p: str(pathlib.Path(p).as_posix()
            .replace('/home/', '/media/')
            .replace('Images/', '')
            .replace('Preprocessed/', '')
            .replace('pngs', '')
            .replace('ChestX-ray8', '')
            )
    )
    df['RelativePath'] = df['Image'].apply(lambda p: str(pathlib.Path(p).relative_to(base_dir)))

    ood = df[df['Dice_RCA_Max'] < threshold].RelativePath
    id = df[df['Dice_RCA_Max'] >= threshold].RelativePath
    return ood.tolist(), id.tolist()
    
def process_file(base_output_dir, rel_path, reduce=True):
    pred_path = pathlib.Path(base_output_dir) / rel_path
    pred_path = pred_path.with_suffix('.csv')
    sigmas = find_avg_std(pred_path)
    return sigmas.mean() if reduce else sigmas


def find_uncertainty_mp(dataset_name, csv_paths, outputs_dir, threshold=0.7, reduce=True, num_workers=8):
    csv_path = csv_paths[dataset_name]
    ood_paths, id_paths = split_distributions(csv_path, threshold=threshold)

    base_output_dir = os.path.join(outputs_dir, dataset_name)
    uncertainty = {}

    for group_name, paths in [('id', id_paths), ('ood', ood_paths)]:
        with Pool(num_workers) as pool:
            process = partial(process_file, base_output_dir, reduce=reduce)
            results = list(tqdm(
                pool.imap_unordered(process, paths),
                total=len(paths),
                desc=f'MP {group_name.upper()} - {dataset_name}'
            ))
            uncertainty[group_name] = np.array(results).flatten()

    return uncertainty

def read_single_latents_file(file_info):
    parent, dataset_name, outputs_dir, intersection, id_parents, ood_parents, filename_to_group = file_info
    latents_local = {'id': [], 'ood': []}
    latents_file = os.path.join(outputs_dir, dataset_name, parent, 'latents.txt')
    if not os.path.exists(latents_file):
        return latents_local

    with open(latents_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]
            sigmas = np.sqrt(np.array(parts[1:], dtype=np.float32))

            if parent in intersection:
                group = filename_to_group.get(filename)
                if group:
                    latents_local[group].append(sigmas)
            elif parent in id_parents:
                latents_local['id'].append(sigmas)
            elif parent in ood_parents:
                latents_local['ood'].append(sigmas)
    return latents_local

def read_latents_mp(dataset_name, csv_paths, outputs_dir, threshold=0.7, max_workers=8):
    csv_path = csv_paths[dataset_name]
    ood_paths, id_paths = split_distributions(csv_path, threshold=threshold)

    ood_parents = {pathlib.Path(p).parent for p in ood_paths}
    id_parents = {pathlib.Path(p).parent for p in id_paths}
    all_parents = ood_parents.union(id_parents)
    intersection = ood_parents & id_parents

    filename_to_group = {
        pathlib.Path(p).name: 'id' for p in id_paths
    } | {
        pathlib.Path(p).name: 'ood' for p in ood_paths
    }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for parent in all_parents:
            futures.append(executor.submit(read_single_latents_file, (parent, dataset_name, outputs_dir, intersection, id_parents, ood_parents, filename_to_group)))

        latents_dict = {'id': [], 'ood': []}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Reading latents'):
            res = future.result()
            latents_dict['id'].extend(res['id'])
            latents_dict['ood'].extend(res['ood'])

    latents_dict['id'] = np.array(latents_dict['id'])
    latents_dict['ood'] = np.array(latents_dict['ood'])
    return latents_dict