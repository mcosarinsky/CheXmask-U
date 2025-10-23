import torch
import torch.nn.functional as F
import numpy as np
import argparse
import scipy.sparse as sp
import cv2
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from utils.dataset_for_train import LandmarksDataset, ToTensor
from utils.metrics import hd_landmarks
from utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart, genMatrixesLungs
from models.HybridGNet2IGSC import Hybrid, HybridNoSkip


def dice_score(pred_mask, gt_mask, class_label):
    pred_bin = (pred_mask == class_label).astype(np.uint8)
    gt_bin = (gt_mask == class_label).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    pred_area = pred_bin.sum()
    gt_area = gt_bin.sum()

    if pred_area + gt_area == 0:
        return 1.0  # perfect score if both are empty
    return 2.0 * intersection / (pred_area + gt_area)
    
def getDenseMask(L, H=None):
    img = np.zeros([1024, 1024], dtype='uint8')

    if L is not None:
        L = L.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [L], -1, 1, -1)

    if H is not None:
        H = H.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [H], -1, 2, -1)

    return img

def unnormalize_landmarks(norm_landmarks, size=1024):
    norm_landmarks = np.clip(norm_landmarks, 0, 1)
    return (norm_landmarks * size).astype(np.int32)

def evaluate(model, val_loader_lungs, val_loader_heart):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    mse_total = 0
    hd_total = 0
    dice_total_lungs = 0
    dice_total_heart = 0
    num_batches = 0
    num_heart_batches = 0

    with torch.no_grad():
        # Lungs dataset (binary mask)
        for sample_batched in tqdm(val_loader_lungs, desc="Testing Lungs"):
            image = sample_batched['image'].to(device)
            target = sample_batched['landmarks'].to(device)

            out = model(image)
            if isinstance(out, tuple):
                out = out[0]

            out = out.reshape(-1, 2)
            target = target.reshape(-1, 2)

            target_np = target.cpu().numpy()
            ts = target_np.shape[0]
            out_np = out.cpu().numpy()[:ts]

            # Unnormalize landmarks from [0,1] to pixel coords
            out_pixels = unnormalize_landmarks(out_np, 1024)
            target_pixels = unnormalize_landmarks(target_np, 1024)

            # Generate masks
            pred_mask = getDenseMask(out_pixels)
            gt_mask = getDenseMask(target_pixels)

            # Compute Dice (class 1 = lungs)
            dice_lung = dice_score(pred_mask, gt_mask, class_label=1)
            dice_total_lungs += dice_lung

            # Compute Hausdorff distance
            dist_RL, dist_LL = hd_landmarks(out, target, 1024, False)
            dist = (dist_RL + dist_LL) / 2
            hd_total += dist

            # Compute MSE
            loss_rec = mean_squared_error(out_np, target_np)
            mse_total += loss_rec
            num_batches += 1

        # Heart dataset (multi-class mask: lungs + heart)
        for sample_batched in tqdm(val_loader_heart, desc="Testing Heart"):
            image = sample_batched['image'].to(device)
            target = sample_batched['landmarks'].to(device)

            out = model(image)
            if isinstance(out, tuple):
                out = out[0]

            out = out.reshape(-1, 2)
            target = target.reshape(-1, 2)

            out_np = out.cpu().numpy()
            target_np = target.cpu().numpy()

            # Unnormalize lungs and heart landmarks separately
            L_out_pixels = unnormalize_landmarks(out_np[:94], 1024)
            H_out_pixels = unnormalize_landmarks(out_np[94:], 1024)
            L_gt_pixels = unnormalize_landmarks(target_np[:94], 1024)
            H_gt_pixels = unnormalize_landmarks(target_np[94:], 1024)

            pred_mask = getDenseMask(L_out_pixels, H_out_pixels)
            gt_mask = getDenseMask(L_gt_pixels, H_gt_pixels)

            # Dice for lungs and heart separately
            dice_lung = dice_score(pred_mask, gt_mask, class_label=1)
            dice_heart = dice_score(pred_mask, gt_mask, class_label=2)

            dice_total_lungs += dice_lung
            dice_total_heart += dice_heart

            # Hausdorff distance (lungs + heart)
            dist_RL, dist_LL, dist_H = hd_landmarks(out, target, 1024, True)
            dist = (dist_RL + dist_LL + dist_H) / 3
            hd_total += dist

            loss_rec = mean_squared_error(out_np, target_np)
            mse_total += loss_rec
            num_batches += 1
            num_heart_batches += 1

    avg_mse = mse_total / num_batches
    avg_hd = hd_total / num_batches
    avg_dice_lungs = dice_total_lungs / num_batches
    avg_dice_heart = dice_total_heart / num_heart_batches 

    print("\n--- TEST RESULTS ---")
    print(f"Average MSE (pixel²): {avg_mse * 1024 * 1024:.2f}")
    print(f"Average RMSE (pixels): {np.sqrt(avg_mse) * 1024:.2f}")
    print(f"Average Hausdorff Distance: {avg_hd:.2f}")
    print(f"Average Dice Score (Lungs only): {avg_dice_lungs:.3f}")
    print(f"Average Dice Score Heart: {avg_dice_heart:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument('--skip_connections', action='store_true', help='Enable skip connections')
    args = parser.parse_args()

    # Load image lists
    images_lungs = open("test_images_lungs.txt").read().splitlines()
    images_heart = open("test_images_heart.txt").read().splitlines()

    print(f"Test Lungs: {len(images_lungs)}")
    print(f"Test Heart: {len(images_heart)}")

    # Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f = 32

    config = {
        'filters': [2, f, f, f, f // 2, f // 2, f // 2],
        'latents': 64,
        'inputsize': 1024,
        'device': device,
        'eval_sampling': False,
        'skip_connections': args.skip_connections,
    }

    # Graph matrices
    A, AD, D, U = genMatrixesLungsHeart()
    N1 = A.shape[0]
    N2 = AD.shape[0]
    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]

    A_ = [sp.csc_matrix(x).tocoo() for x in [A, A, A, AD, AD, AD]]
    D_ = [sp.csc_matrix(D).tocoo()]
    U_ = [sp.csc_matrix(U).tocoo()]

    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to(device) for x in X] for X in (A_, D_, U_))

    # Initialize model
    if args.skip_connections:
        model = Hybrid(config, D_t, U_t, A_t)
    else:
        model = HybridNoSkip(config, D_t, U_t, A_t)

    model.to(device)
    print(f"Using {'skip connections' if args.skip_connections else 'no skip connections'}")


    # Load weights
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded model weights from {args.weights}")

    # Downsampling matrix for lungs
    _, _, DL, _ = genMatrixesLungs()
    DL = sp.csc_matrix(DL).tocoo()
    DL_t = [scipy_to_torch_sparse(DL).to(device)]

    # Datasets
    test_dataset_lungs = LandmarksDataset(
        images=images_lungs,
        img_path="../X-Ray/Chest-xray-landmark-dataset/Images",
        label_path="../X-Ray/Chest-xray-landmark-dataset/landmarks",
        organ='L',
        transform=ToTensor()
    )

    test_dataset_heart = LandmarksDataset(
        images=images_heart,
        img_path="../X-Ray/Chest-xray-landmark-dataset/Images",
        label_path="../X-Ray/Chest-xray-landmark-dataset/landmarks",
        organ='LH',
        transform=ToTensor()
    )

    test_loader_lungs = torch.utils.data.DataLoader(test_dataset_lungs, batch_size=1, shuffle=False)
    test_loader_heart = torch.utils.data.DataLoader(test_dataset_heart, batch_size=1, shuffle=False)

    # Run evaluation
    evaluate(model, test_loader_lungs, test_loader_heart)
