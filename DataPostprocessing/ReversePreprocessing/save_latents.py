#!/usr/bin/env python3
"""
Script to read latents.txt files for multiple datasets, convert to CSVs,
and merge them into the dataset CSVs (preprocessed version).
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pathlib
import argparse
from pathlib import Path

# -----------------------------
# Latents parsing
# -----------------------------
def read_single_latents_file(latents_file):
    rows = []
    if not os.path.exists(latents_file):
        return rows

    folder = os.path.dirname(latents_file)
    with open(latents_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[0]
            sigmas = np.sqrt(np.array(parts[1:], dtype=np.float32))
            image_path = os.path.join(folder, filename)
            rows.append((image_path, sigmas))
    return rows


def read_latents_dataframe(dataset_name, outputs_dir, max_workers=8):
    dataset_dir = os.path.join(outputs_dir, dataset_name)
    latents_files = list(pathlib.Path(dataset_dir).rglob("latents.txt"))

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_single_latents_file, str(f)) for f in latents_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Reading latents ({dataset_name})"):
            rows.extend(future.result())

    # Convert latents arrays to comma-separated strings rounded to 4 decimals
    df = pd.DataFrame(rows, columns=["image_path", "latents"])
    df["latents"] = df["latents"].apply(lambda x: ",".join([f"{v:.4f}" for v in x]))
    return df


# -----------------------------
# Merge with dataset CSVs
# -----------------------------
def merge_latents_into_dataset(dataset_name, df_latents, dataset_csv_path):
    df_csv = pd.read_csv(dataset_csv_path)

    # Build merge key depending on dataset
    if dataset_name == 'CheXpert':
        def last_3_parents_plus_stem(p):
            p = Path(p)
            parts = list(p.parts)
            key = Path(*parts[-4:-1], parts[-1].split(".")[0]).as_posix()
            return key

        df_latents['merge_key'] = df_latents['image_path'].apply(last_3_parents_plus_stem)
        df_csv['merge_key'] = df_csv[df_csv.columns[0]].apply(last_3_parents_plus_stem)
    else:
        df_latents['merge_key'] = df_latents['image_path'].apply(lambda x: Path(x).stem)
        df_csv['merge_key'] = df_csv[df_csv.columns[0]].apply(lambda x: Path(x).stem)

    # Merge on key
    df_merged = df_csv.merge(
        df_latents[['merge_key', 'latents']],
        on='merge_key',
        how='left'
    )

    df_merged = df_merged.drop(columns=['merge_key'], errors='ignore')
    return df_merged


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert latents.txt to CSVs and merge into dataset CSVs")
    parser.add_argument("--outputs_dir", type=str, default="Outputs/Predictions/CheXmask",
                        help="Folder containing dataset folders with latents.txt files")
    parser.add_argument("--latents_folder", type=str, default="LatentsData",
                        help="Folder to save intermediate latents CSVs")
    parser.add_argument("--datasets", nargs="+",
                        default=["VinDr", "Padchest", "CheXpert", "MIMIC", "ChestX-ray8"],
                        help="List of datasets to process")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of threads to use when reading files")
    args = parser.parse_args()

    os.makedirs(args.latents_folder, exist_ok=True)

    base_pre = "/media/ngaggion/DATA/chexmask-database/Preprocessed"
    base_orig = "/media/ngaggion/DATA/chexmask-database/OriginalResolution"
    base_folder_chest8 = "/media/ngaggion/DATA/chexmask-database/OriginalResolution"

    datasets_pre = {
        'VinDr': os.path.join(base_pre, 'VinDr-CXR.csv'),
        'Padchest': os.path.join(base_pre, 'Padchest.csv'),
        'CheXpert': os.path.join(base_pre, 'CheXpert.csv'),
        'MIMIC': os.path.join(base_pre, 'MIMIC-CXR-JPG.csv'),
        'ChestX-ray8': os.path.join(base_folder_chest8, 'ChestX-Ray8.csv'),
    }

    datasets_orig = {
        #'VinDr': os.path.join(base_orig, 'VinDr-CXR.csv'),
        #'Padchest': os.path.join(base_orig, 'Padchest.csv'),
        #'CheXpert': os.path.join(base_orig, 'CheXpert.csv'),
        'MIMIC': os.path.join(base_orig, 'MIMIC-CXR-JPG.csv'),
    }

    for dataset in args.datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Step 1: read latents into DataFrame
        #df_latents = read_latents_dataframe(dataset, args.outputs_dir, max_workers=args.max_workers)

        # Step 2: save standalone latents CSV
        #latents_csv_path = os.path.join(args.latents_folder, f"{dataset}.csv")
        #df_latents.to_csv(latents_csv_path, index=False)
        #print(f"Saved {dataset} latents CSV -> {latents_csv_path}")

        df_latents = pd.read_csv(os.path.join(args.latents_folder, f"{dataset}.csv"))
        
        # Step 3: merge with dataset CSV
        if dataset in datasets_orig:
            df_merged_orig = merge_latents_into_dataset(dataset, df_latents, datasets_orig[dataset])
            df_merged_orig.to_csv(datasets_orig[dataset], index=False)
            print(f"Updated {datasets_orig[dataset]} with latents.")

        #if dataset in datasets_pre:
        #    df_merged_pre = merge_latents_into_dataset(dataset, df_latents, datasets_pre[dataset])
        #    df_merged_pre.to_csv(datasets_pre[dataset], index=False)
        #    print(f"Updated {datasets_pre[dataset]} with latents.")

if __name__ == "__main__":
    main()
