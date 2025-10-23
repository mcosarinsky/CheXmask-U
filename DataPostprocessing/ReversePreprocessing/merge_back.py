import os
import pandas as pd


def process_and_save_chunks(datasets_rca, datasets_pre_or_orig, chunk_size=20000):
    for dataset in datasets_pre_or_orig.keys():
        print(f"\nProcessing {dataset} in chunks...")
        df_1 = pd.read_csv(datasets_rca[dataset])

        # Rename columns in df_1
        df_1 = df_1.rename(columns={
            'Dice_RCA_Max': 'Dice RCA (Max)',
            'Dice_RCA_Mean': 'Dice RCA (Mean)'
        })
        
        #df_1['Image'] = df_1['Image'] + '.png'
        # Prepare output file path
        orig_path = datasets_pre_or_orig[dataset]
        dir_name = os.path.dirname(orig_path)
        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        new_path = os.path.join(dir_name, f"{base_name}_v2.csv")
        
        # Delete old output if exists
        if os.path.exists(new_path):
            os.remove(new_path)
        
        # Iterate over df_2 in chunks
        for i, chunk in enumerate(pd.read_csv(orig_path, chunksize=chunk_size)):
            first_col = chunk.columns[1]

            # Take the same slice from df_1
            start = i * chunk_size
            end = start + len(chunk)
            df_1_slice = df_1['Image'].iloc[start:end]
            
            if not df_1_slice.reset_index(drop=True).equals(chunk[first_col].reset_index(drop=True)):
                print(f"Chunk {i}: No exact match, skipping entire dataset {dataset}.")
                break
            else:
                chunk['Dice RCA (Mean)'] = df_1['Dice RCA (Mean)'].iloc[start:end].values
                chunk['Dice RCA (Max)'] = df_1['Dice RCA (Max)'].iloc[start:end].values

                # Step 4: Reorder columns
                final_cols = [first_col, 'Dice RCA (Mean)', 'Dice RCA (Max)', 
                            'Landmarks (Mean)', 'Landmarks (Std)', 'Left Lung', 
                            'Right Lung', 'Heart', 'Height', 'Width']
                chunk = chunk[[c for c in final_cols if c in chunk.columns]]

                # Step 5: Append chunk to CSV
                chunk.to_csv(new_path, mode='a', index=False, header=(i==0))
                print(f"Chunk {i} processed and saved.")
            
        
        print(f"Finished processing {dataset}. Saved to {new_path}")

# Define your base paths (no changes needed here)
base_pre = "/media/ngaggion/DATA/chexmask-database/Preprocessed"
base_orig = "/media/ngaggion/DATA/chexmask-database/OriginalResolution"
base_folder_chest8 = "/media/ngaggion/DATA/chexmask-database/OriginalResolution"
base_rca = "/media/ngaggion/DATA/chexmask-database"

# Datasets
datasets_pre = {
    'VinDr': os.path.join(base_pre, 'VinDr-CXR.csv'),
    'Padchest': os.path.join(base_pre, 'Padchest.csv'),
    'CheXpert': os.path.join(base_pre, 'CheXpert.csv'),
    'MIMIC': os.path.join(base_pre, 'MIMIC-CXR-JPG.csv'),
    'ChestX-ray8': os.path.join(base_folder_chest8, 'ChestX-Ray8.csv'),
}   

datasets_orig = {
    'VinDr': os.path.join(base_orig, 'VinDr-CXR.csv'),
    'Padchest': os.path.join(base_orig, 'Padchest.csv'),
    'CheXpert': os.path.join(base_orig, 'CheXpert.csv'),
    'MIMIC': os.path.join(base_orig, 'MIMIC-CXR-JPG.csv'),
}

datasets_rca = {
    'VinDr': os.path.join(base_rca, 'VinBigData_RCA.csv'),
    'Padchest': os.path.join(base_rca, 'Padchest_RCA.csv'),
    'CheXpert': os.path.join(base_rca, 'CHEXPERT_RCA.csv'),
    'MIMIC': os.path.join(base_rca, 'MIMIC_RCA.csv'),
    'ChestX-ray8': os.path.join(base_rca, 'ChestX-ray8_RCA.csv'),
}

# Now apply the function to both datasets
process_and_save_chunks(datasets_rca, datasets_pre)
process_and_save_chunks(datasets_rca, datasets_orig)
