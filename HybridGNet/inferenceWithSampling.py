import os
import cv2
import torch
import re
import numpy as np
import argparse
import pathlib
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm
from models.HybridGNet2IGSC import Hybrid, HybridNoSkip
from utils.utils import genMatrixesLungsHeart, scipy_to_torch_sparse


def main(n_samples, image_folder, output_folder, weights_path, skip_connections):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and prepare matrices
    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]

    config = {
        'n_nodes': [N1, N1, N1, N2, N2, N2],
        'latents': 64,
        'inputsize': 1024,
        'K': 6,
        'filters': [2, 32, 32, 32, 16, 16, 16],
        'skip_features': 32,
        'eval_sampling': True
    }

    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

    # Load model
    if skip_connections:
        print("Using skip connections")
        model = Hybrid(config, D_t, U_t, A_t).to(device)
    else:
        print("Using no skip connections")
        model = HybridNoSkip(config, D_t, U_t, A_t).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("Model loaded")

    # Recursively find images
    image_folder = pathlib.Path(image_folder)
    output_folder = pathlib.Path(output_folder)
    valid_exts = [".png", ".jpg"]
    image_paths = [p for p in image_folder.rglob("*") if p.suffix.lower() in valid_exts]
    print(f"Processing {len(image_paths)} images from {image_folder}...")

    latents_per_folder = defaultdict(list)

    with torch.no_grad():
        for image_path in tqdm(image_paths):
            # Read image
            image = cv2.imread(str(image_path), 0) / 255.0
            data = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device).float()

            # Encode once
            mu, log_var, conv6, conv5 = model.encode(data)
            latent_var = np.exp(log_var.cpu().numpy())

            # Determine relative path and create output folder
            relative_path = image_path.relative_to(image_folder).parent
            target_dir = output_folder / relative_path
            target_dir.mkdir(parents=True, exist_ok=True)

            # Save latent var under corresponding folder
            latents_per_folder[target_dir].append((image_path.name, latent_var.flatten()))

            # Sample N latent vectors to decode
            zs = [model.sampling(mu, log_var) for _ in range(n_samples)]
            z_exp = torch.stack(zs, dim=0)

            # Expand skip connections to match batch size
            conv6_exp = conv6.repeat(n_samples, 1, 1, 1)
            conv5_exp = conv5.repeat(n_samples, 1, 1, 1)

            # Decode in batch
            batch_output, _, _ = model.decode(z_exp, conv6_exp, conv5_exp)
            batch_output = batch_output.cpu().numpy().reshape(n_samples, -1, 2) * 1024  # scale back

            # Compute mean and std per node
            means = np.mean(batch_output, axis=0)
            stds = np.std(batch_output, axis=0)

            # Save stats to CSV
            output_csv_path = target_dir / (image_path.stem + ".csv")
            with open(output_csv_path, "w") as f:
                f.write("Node,Mean x,Mean y,Std x,Std y\n")
                for i, (m, s) in enumerate(zip(means, stds)):
                    f.write(f"{i},{m[0]},{m[1]},{s[0]},{s[1]}\n")

    # After loop: write one latents.txt per folder
    for folder, latents in latents_per_folder.items():
        latents_path = folder / "latents.txt"
        with open(latents_path, "w") as f:
            for name, var in latents:
                var_str = " ".join(map(str, var))
                f.write(f"{name} {var_str}\n")

    print(f"All results saved under: {output_folder}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample multiple predictions from a trained model.")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples per image.")
    parser.add_argument("--image_folder", type=str, default="", help="Subfolder inside base_folder for images (optional).")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to model weights (.pt file).")
    parser.add_argument("--skip_connections", action="store_true", help="Use skip connections in the model.")

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    base_folder = "Outputs"
    subfolder = "Skip" if args.skip_connections else "NoSkip"

    image_folder = os.path.join(base_folder, "Images/FinalDatabase/Images")
    output_folder = os.path.join(base_folder, "Predictions/FinalDatabase")
    #image_folder = os.path.join(base_folder, "Images", args.image_folder)
    #output_folder = os.path.join(base_folder, "Predictions", subfolder, args.image_folder)

    #main(args.n_samples, image_folder, output_folder, args.weights_path, args.skip_connections)
    
    #image_folder = "../X-Ray/BigScale/Datasets/CheXpert/Preprocessed/valid/"
    #output_folder = os.path.join(base_folder, "Predictions/CheXmask", "CheXpert/valid")
    #image_folder = "../X-Ray/BigScale/Datasets/MIMIC/Images/"
    #output_folder = os.path.join(base_folder, "Predictions/CheXmask", "MIMIC")
    #image_folder = "../X-Ray/BigScale/Datasets/Padchest/Images/"
    #output_folder = os.path.join(base_folder, "Predictions/CheXmask", "Padchest")
    #image_folder = "../X-Ray/BigScale/Datasets/VinBigData/pngs/"
    #output_folder = os.path.join(base_folder, "Predictions/CheXmask", "VinDr")

    #folder = "../X-Ray/BigScale/Datasets/Chest8/ChestX-ray8/"
    #for f in os.listdir(folder):
    #    # check if it is a subfolder
    #    if os.path.isdir(os.path.join(folder, f)):
    #        image_folder = os.path.join(folder, f)
    #        output_folder = os.path.join(base_folder, "Predictions/CheXmask", "ChestX-ray8", f)
    #        main(args.n_samples, image_folder, output_folder, args.weights_path, args.skip_connections)
    
    #python HybridGNet/inferenceWithSampling.py --skip_connections --weights "Weights/Hybrid_LH_FULL/bestMSE.pt"

    main(args.n_samples, image_folder, output_folder, args.weights_path, args.skip_connections)