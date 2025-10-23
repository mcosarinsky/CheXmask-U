import cv2
import numpy as np
import os
import random

def apply_occlusion(image: np.ndarray, landmarks: list, size: int) -> np.ndarray:
    """
    Apply occlusion around specified landmarks.

    Parameters:
      image (np.ndarray): Grayscale image.
      landmarks (list): List of (x, y) tuples.
      size (int): Size of the square occlusion region.

    Returns:
      np.ndarray: Occluded image (uint8).
    """
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")

    occluded_image = image.copy()
    for x, y in landmarks:
        x_start = max(0, x - size)
        x_end = min(image.shape[1], x + size + 1)
        y_start = max(0, y - size)
        y_end = min(image.shape[0], y + size + 1)
        occluded_image[y_start:y_end, x_start:x_end] = 0
    return occluded_image.astype(np.uint8)

def apply_blur(image: np.ndarray, landmarks: list, size: int, severity: float) -> np.ndarray:
    """
    Apply Gaussian blur around specified landmarks.

    Parameters:
      image (np.ndarray): Grayscale image.
      landmarks (list): List of (x, y) tuples.
      size (int): Size of the square region.
      severity (float): Blur severity (0 to 1).

    Returns:
      np.ndarray: Blurred image (uint8).
    """
    if severity < 0:
        raise ValueError("severity must be non-negative")
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")

    blurred_image = image.astype(np.float32).copy()
    for x, y in landmarks:
        x_start = max(0, x - size)
        x_end = min(image.shape[1], x + size + 1)
        y_start = max(0, y - size)
        y_end = min(image.shape[0], y + size + 1)
        region = blurred_image[y_start:y_end, x_start:x_end]
        ksize = int(2 * severity + 1)  # Kernel size must be odd
        if ksize > 1:
            region = cv2.GaussianBlur(region, (ksize, ksize), 0)
        region = np.clip(region, 0, 255)
        blurred_image[y_start:y_end, x_start:x_end] = region
    return blurred_image.astype(np.uint8)

def blur(image, severity, type='gaussian'):
    if severity < 1:
        return image  
    
    kernel_size = 2*severity + 1 # Define an odd kernel size based on severity

    if type == 'gaussian':
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif type == 'median':
        blurred_image = cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError("Invalid blur type. Must be 'gaussian' or 'median'.")
    
    return blurred_image

def gaussian_noise(image, severity):
    noise = np.random.normal(0, severity, image.shape)  

    # Add noise to the image and clip values
    noisy_image = image.astype(np.float32) + noise 
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def apply_corruption(img_dir: str, output_subdir: str, corruption_fn, severity_levels: list, n_samples=15, **kwargs):
    output_dir = os.path.join(img_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    selected_images = random.sample(all_images, min(n_samples, len(all_images)))
    
    for img_name in selected_images:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_base, ext = os.path.splitext(img_name)
        
        for severity in severity_levels:
            processed_img = corruption_fn(img, severity, **kwargs)
            output_name = f"{img_base}_{severity}{ext}"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, processed_img)

def blend_images(datasets, N_samples, output_dir, alpha_step=0.05):
    """
    For a given number of samples, randomly picks 2 images from the same dataset (without repeating combinations)
    and blends them for alpha values from 0 to 1 (step alpha_step). Save blended images with filenames reflecting the alpha level.
    
    Parameters:
        datasets (dict): Dictionary with keys indicating dataset and list of image file paths as values.
        N_samples (int): Number of unique image combinations to process.
        output_dir (str): Directory where the blended images will be saved.
        alpha_step (float): Step size for alpha from 0 to 1.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    used_combinations = {key: set() for key in datasets}  # track used pairs for each dataset
    
    sample_count = 0
    keys = list(datasets.keys())
    while sample_count < N_samples:
        # Randomly pick a dataset 
        key = random.choice(keys)

        # Randomly select two distinct images
        img_pair = tuple(sorted(random.sample(datasets[key], 2)))
        # Check if this combination (order doesn't matter) has been processed already
        if img_pair in used_combinations[key]:
            continue
        
        # Mark the combination as used
        used_combinations[key].add(img_pair)
        sample_count += 1
        
        # Read the two images
        img1 = cv2.imread(img_pair[0], cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(img_pair[1], cv2.IMREAD_UNCHANGED)
        
        # Get base names for file naming (without extension)
        base1 = os.path.splitext(os.path.basename(img_pair[0]))[0]
        base2 = os.path.splitext(os.path.basename(img_pair[1]))[0]
        
        # For each alpha value, blend and save
        alphas = np.arange(0, 1 + alpha_step, alpha_step)
        for alpha in alphas:
            blended =  cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
            alpha_int = int(round(alpha * 100))

            # Create filename with image names and alpha level
            filename = f"{base1}_{base2}_{alpha_int}.png"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, blended)