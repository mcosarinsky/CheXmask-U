#!/usr/bin/env python3
"""
Full Dataset Processing Pipeline
1. Load RCA landmark predictions
2. Merge with dataset CSVs
3. Convert Mean/Std coordinates into string format
4. Generate RLE masks for RL/LL/Heart
5. Save updated CSVs
"""

import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from joblib import Parallel, delayed

# ---------------- Utils ---------------- #

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

# ---------------- Step 1: Load predictions ---------------- #

def _read_full_prediction_full(base_output_dir, rel):
    pred_path = (Path(base_output_dir) / rel).with_suffix(".csv")
    if not pred_path.exists():
        return None
    try:
        pred_df = pd.read_csv(pred_path)
        col_arrays = {col: pred_df[col].to_numpy() for col in pred_df.columns}
        return col_arrays
    except Exception:
        return None

def build_df_with_preds(dataset_name, csv_paths, outputs_dir, num_workers=8):
    csv_path = Path(csv_paths[dataset_name]).resolve()
    base_dir = csv_path.parent
    df = pd.read_csv(csv_path)

    df["OriginalImage"] = df["Image"]
    df["Image"] = df["Image"].apply(
        lambda p: str(
            Path(p).as_posix()
            .replace("/home/", "/media/")
            .replace("Images/", "")
            .replace("Preprocessed/", "")
            .replace("pngs", "")
            .replace("ChestX-ray8", "")
        ).replace("//", "/")
    )

    df["RelativePath"] = df["Image"].apply(lambda p: str(Path(p).relative_to(base_dir)))
    base_output_dir = Path(outputs_dir) / dataset_name
    rel_paths = df["RelativePath"].tolist()

    results = [None] * len(rel_paths)
    with Pool(num_workers) as pool:
        for i, val in enumerate(
            tqdm(
                pool.imap(partial(_read_full_prediction_full, base_output_dir), rel_paths),
                total=len(rel_paths),
                desc=f"Load full preds - {dataset_name}",
            )
        ):
            results[i] = val

    if any(r is not None for r in results):
        all_cols = [col for r in results if r is not None for col in r.keys()]
        all_cols = sorted(set(all_cols))
        for col in all_cols:
            df[col] = [r[col] if r is not None else np.array([]) for r in results]

    if dataset_name == "ChestX-ray8":
        df["RelativePath"] = df["RelativePath"].apply(lambda p: Path(p).name)

    return df

# ---------------- Step 3: Convert Mean/Std ---------------- #

def process_mean_std_columns(df):
    def list_to_int_str(xs, ys):
        return ",".join([f"{int(round(x))},{int(round(y))}" for x, y in zip(xs, ys)])
    def list_to_float_str(xs, ys):
        return ",".join([f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys)])

    if all(col in df.columns for col in ["Mean x", "Mean y"]):
        df["Mean"] = df.apply(lambda r: list_to_int_str(r["Mean x"], r["Mean y"]), axis=1)
        df.drop(columns=["Mean x", "Mean y"], inplace=True)

    if all(col in df.columns for col in ["Std x", "Std y"]):
        df["Std"] = df.apply(lambda r: list_to_float_str(r["Std x"], r["Std y"]), axis=1)
        df.drop(columns=["Std x", "Std y"], inplace=True)

    return df

# ---------------- Step 4: Generate RLE and update Mean/Std ---------------- #

def process_row_orig(row, preproc_size=1024):
    try:
        h, w = int(row.Height), int(row.Width)
        if pd.isna(row.Mean):
            return {"RL_Mean": None, "LL_Mean": None, "H_Mean": None,
                    "Mean": row.Mean, "Std": row.Std}

        mean_vals, mean_str_out, std_str_out = remap_landmarks_and_std(row.Mean, row.Std, h, w, preproc_size)
        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]
        return {
            "RL_Mean": get_RLE_from_mask(getDenseMask(RL, h, w)),
            "LL_Mean": get_RLE_from_mask(getDenseMask(LL, h, w)),
            "H_Mean":  get_RLE_from_mask(getDenseMask(H, h, w)),
            "Mean": mean_str_out,
            "Std": std_str_out
        }
    except Exception as e:
        print(f"Error processing row {getattr(row, 'Image', 'unknown')}: {e}")
        return {"RL_Mean": None, "LL_Mean": None, "H_Mean": None,
                "Mean": row.Mean, "Std": row.Std}

def process_row_pre(row):
    try:
        h, w = int(row.Height), int(row.Width)
        if pd.isna(row.Mean):
            return {"RL_Mean": None, "LL_Mean": None, "H_Mean": None}

        mean_vals = np.array([int(v) for v in row.Mean.split(",")]).reshape(-1, 2)
        RL, LL, H = mean_vals[:44], mean_vals[44:94], mean_vals[94:]

        return {
            "RL_Mean": get_RLE_from_mask(getDenseMask(RL, h, w)),
            "LL_Mean": get_RLE_from_mask(getDenseMask(LL, h, w)),
            "H_Mean":  get_RLE_from_mask(getDenseMask(H, h, w)),
        }
    except Exception as e:
        print(f"Error processing row {getattr(row, 'Image', 'unknown')}: {e}")
        return {"RL_Mean": None, "LL_Mean": None, "H_Mean": None}


def process_df_parallel(df, func, n_jobs=-1, chunk_size=500):
    rows = list(df.itertuples(index=False))
    all_results = []
    with tqdm(total=len(rows), desc="RLE masks", unit="row") as pbar:
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i:i+chunk_size]
            chunk_results = Parallel(n_jobs=n_jobs)(delayed(func)(row) for row in chunk)
            all_results.extend(chunk_results)
            pbar.update(len(chunk))
    return pd.DataFrame(all_results)

# ---------------- Step 5: Main ---------------- #

if __name__ == "__main__":
    csv_parents = "../X-Ray/BigScale/Datasets/"
    outputs_dir = "Outputs/Predictions/CheXmask/"
    base_seg_folder = "/media/ngaggion/DATA/chexmask-database/OriginalResolution"
    base_pre_folder = "/media/ngaggion/DATA/chexmask-database/Preprocessed"

    csv_paths = {
        "MIMIC": os.path.join(csv_parents, "MIMIC/MIMIC_clean_RCA.csv"),
        "VinDr": os.path.join(csv_parents, "VinBigData/VinBigData_RCA.csv"),
        "Padchest": os.path.join(csv_parents, "Padchest/Padchest_clean_RCA.csv"),
        "CheXpert": os.path.join(csv_parents, "CheXpert/CHEXPERT_RCA.csv"),
        "ChestX-ray8": os.path.join(csv_parents, "ChestX-ray8/ChestX-ray8_RCA.csv"),
    }

    datasets_orig = {
        "VinDr": os.path.join(base_seg_folder, "VinDr-CXR.csv"),
        "Padchest": os.path.join(base_seg_folder, "Padchest.csv"),
        "CheXpert": os.path.join(base_seg_folder, "CheXpert.csv"),
        "MIMIC": os.path.join(base_seg_folder, "MIMIC-CXR-JPG.csv"),
    }

    datasets_pre = {
        "VinDr": os.path.join(base_pre_folder, "VinDr-CXR.csv"),
        "Padchest": os.path.join(base_pre_folder, "Padchest.csv"),
        "CheXpert": os.path.join(base_pre_folder, "CheXpert.csv"),
        "MIMIC": os.path.join(base_pre_folder, "MIMIC-CXR-JPG.csv"),
        "ChestX-ray8": os.path.join(base_pre_folder, "ChestX-Ray8.csv"),
    }


    for dataset_name in csv_paths.keys():
        print(f"\nProcessing dataset: {dataset_name}")

        # Step 1: Load predictions
        df_preds = build_df_with_preds(dataset_name, csv_paths, outputs_dir)

        # Choose configs
        for kind, datasets_cfg, row_func in [
            ("orig", datasets_orig, process_row_orig),
            ("pre", datasets_pre, process_row_pre),
        ]:
            if dataset_name not in datasets_cfg:
                continue

            print(f" → Processing {dataset_name} ({kind})")
            df_csv = pd.read_csv(datasets_cfg[dataset_name])

            # Drop 'Avg std' if exists
            if "Avg std" in df_csv.columns:
                df_csv = df_csv.drop(columns=["Avg std"])

            drop_cols = [
                "Unnamed: 0", "Dice_RCA_Max", "Dice_RCA_Mean",
                "L_Dice_RCA_Max", "L_Dice_RCA_Mean", "H_Dice_RCA_Max",
                "H_Dice_RCA_Mean", "OriginalImage", "RelativePath", "Node",
                "RL_Mean", "LL_Mean", "H_Mean"
            ]
            df_other = df_preds.drop(columns=[c for c in drop_cols if c in df_preds], errors="ignore").copy()

            # Merge keys
            if dataset_name == "CheXpert":
                def last_3_parents_plus_stem(p):
                    p = Path(p)
                    parts = list(p.parts)
                    return Path(*parts[-4:-1], p.stem).as_posix()
                df_other["merge_key"] = df_other["Image"].apply(last_3_parents_plus_stem)
                df_csv["merge_key"] = df_csv[df_csv.columns[0]].apply(last_3_parents_plus_stem)
            else:
                df_other["merge_key"] = df_other["Image"].apply(lambda x: Path(x).stem)
                df_csv["merge_key"] = df_csv[df_csv.columns[0]].apply(lambda x: Path(x).stem)

            df_merged = df_csv.merge(df_other, on="merge_key", how="left")
            df_merged = df_merged.drop(columns=["merge_key", "Image"], errors="ignore")

            # Step 3: Convert coords & generate RLE
            df_merged = process_mean_std_columns(df_merged)

            # Generate RLE and update Mean/Std (only for orig)
            df_rle = process_df_parallel(df_merged, row_func)
            df_merged = pd.concat([df_merged, df_rle], axis=1)

            # Save updated CSV
            out_path = datasets_cfg[dataset_name]
            df_merged.to_csv(out_path, index=False)
            print(f"✓ Saved {dataset_name} ({kind}) to {out_path}")

