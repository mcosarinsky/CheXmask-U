import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score


def plot_mean_with_color_gradient(df: pd.DataFrame, img=None, fig=None, ax=None, show_bar: bool = True, vmin=0, vmax=None):
    """
    Plot the mean (x, y) for each node with a color gradient representing uncertainty.

    Parameters:
      df (pd.DataFrame): DataFrame with columns ['Mean x', 'Mean y', 'Std x', 'Std y'], indexed by node.
      img: Optional background image.
      fig, ax: Matplotlib figure and axes objects.
      show_bar (bool): Whether to display a colorbar.
      vmin, vmax: Color mapping limits.

    Returns:
      fig, ax, scatter: The figure, axes, and scatter plot object.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    if img is not None:
        ax.imshow(img, cmap='gray')

    # Average of std x and std y for color
    sigma_avg = (df['Std x'] + df['Std y']) / 2

    scatter = ax.scatter(df['Mean x'], df['Mean y'], c=sigma_avg, cmap='hot', s=50, vmin=vmin, vmax=vmax)

    if show_bar:
        fig.colorbar(scatter, ax=ax, label='Average Std (Uncertainty)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(0, 1024)
    ax.set_ylim(0, 1024)
    ax.invert_yaxis()

    return fig, ax, scatter

def plot_relative_uncertainty(df_orig, df_corr, img_corr):
    """
    Plot the ratio of uncertainties (corr/orig) at each node on the corrupted image.
    Assumes both dataframes have ['Mean x', 'Mean y', 'Std x', 'Std y'] columns and node index.
    """
    # Compute average uncertainty per node
    sigma_o = (df_orig['Std x'] + df_orig['Std y']) / 2
    sigma_c = (df_corr['Std x'] + df_corr['Std y']) / 2
    ratio = (sigma_c / sigma_o).rename("ratio")

    # Get corrupted means for plotting
    means_c = df_corr[['Mean x', 'Mean y']]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_corr, cmap='gray')
    scatter = ax.scatter(
        means_c['Mean x'], means_c['Mean y'],
        c=ratio.values, cmap='hot', s=50
    )
    ax.set_title("Relative Uncertainty (σ_corr / σ_orig)")
    ax.set_xlim(0, img_corr.shape[1])
    ax.set_ylim(img_corr.shape[0], 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax, scatter

def plot_comparison(df_orig, img_orig, df_corr, img_corr, plot_fn, show_global_bar=False, **kwargs):
    """
    Create a side-by-side comparison plot using the provided plotting function.

    Parameters:
      df_orig (pd.DataFrame): DataFrame for the original image.
      img_orig (np.ndarray): Original image.
      df_corr (pd.DataFrame): DataFrame for the corrupted image.
      img_corr (np.ndarray): Corrupted image.
      plot_fn (callable): Plotting function to use (e.g., plot_mean_with_color_gradient or plot_mean_with_uncertainty).
    
    Returns:
      tuple: (fig, (ax1, ax2))
    """
    # Create figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for original image
    fig, ax1, scatter1 = plot_fn(df_orig, img=img_orig, fig=fig, ax=ax1, **kwargs)
    #ax1.set_title('Original', fontsize=16)

    # Plot for corrupted image
    fig, ax2, scatter2 = plot_fn(df_corr, img=img_corr, fig=fig, ax=ax2, **kwargs)
    ax2.set_title('Corrupted', fontsize=16)

    # If 'show_bar' is specified and True, add a shared colorbar
    if show_global_bar:
        cbar = fig.colorbar(scatter1, ax=[ax1, ax2], orientation='vertical', fraction=0.05)
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        cbar.ax.set_position([0.87, 0.15, 0.03, 0.7])

    return fig, (ax1, ax2)

def plot_sigmas_mean(sigma_dict, yscale='linear', grid=True):
    """Plots sigma values for a given sigma dictionary and returns fig, ax."""
    n_images = len(sigma_dict)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, (img_name, vals) in enumerate(sigma_dict.items()):
        ax = axes[i]
        ax.scatter(vals['corr_levels'], np.array(vals['sigmas']).mean(axis=1))
        ax.set_title(f"{img_name[:20]}")
        ax.set_yscale(yscale)
        if grid: 
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig, axes  # Return fig, axes instead of showing it

def plot_global_uncertainty(sigma_dict, title='', ax=None):
    """Plots global uncertainty using a boxplot and returns fig, ax."""
    data = []
    labels = []

    for key, value in sigma_dict.items():
        data.append(np.array(value).flatten())
        labels.append(key)

    # If ax is provided, plot on the existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure  # If ax is passed, we use the provided figure

    sns.boxplot(data=data, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Corruption', fontsize=13)
    ax.set_ylabel('Sigma', fontsize=13)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    return fig, ax

def plot_kde_comparison_grid(
    id_skip, ood_skip, id_noskip, ood_noskip, datasets, 
    clip_skip=(0, 100), clip_noskip=(0, 100),
    n_cols=5, height_per_row=6):
    
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, height_per_row * n_rows))
    axes = axes.reshape(n_rows, n_cols)

    for i, dataset in enumerate(datasets):
        col = i % n_cols

        # Flexible clip handling
        clip_skip_val = clip_skip[dataset] if isinstance(clip_skip, dict) else clip_skip
        clip_noskip_val = clip_noskip[dataset] if isinstance(clip_noskip, dict) else clip_noskip

        # --- Skip row ---
        ax_skip = axes[0, col]
        sns.kdeplot(ood_skip[dataset], fill=True, color="red", label="OOD", ax=ax_skip, alpha=0.5, clip=clip_skip_val)
        sns.kdeplot(id_skip[dataset], fill=True, color="blue", label="ID", ax=ax_skip, alpha=0.5, clip=clip_skip_val)
        ax_skip.set_xlim(left=0)
        if col == 0:
            ax_skip.set_ylabel("Density", fontsize=17)
        else:
            ax_skip.set_ylabel("")  # only first column has ylabel
        ax_skip.set_xlabel("")  # only bottom row has xlabel
        ax_skip.set_title(dataset, fontsize=19)
        ax_skip.legend(fontsize=13)
        ax_skip.tick_params(axis='both', labelsize=14)

        # --- No-skip row ---
        ax_noskip = axes[1, col]
        sns.kdeplot(ood_noskip[dataset], fill=True, color="red", label="OOD", ax=ax_noskip, alpha=0.5, clip=clip_noskip_val)
        sns.kdeplot(id_noskip[dataset], fill=True, color="blue", label="ID", ax=ax_noskip, alpha=0.5, clip=clip_noskip_val)
        ax_noskip.set_xlim(left=0)
        if col == 0:
            ax_noskip.set_ylabel("Density", fontsize=17)
        else:
            ax_noskip.set_ylabel("")  # only first column has ylabel
        ax_noskip.set_xlabel("Sigma", fontsize=17)
        ax_noskip.legend(fontsize=13)
        ax_noskip.tick_params(axis='both', labelsize=14)

    # Match y-limits per row
    y_skip_max = max(ax.get_ylim()[1] for ax in axes[0, :len(datasets)])
    y_noskip_max = max(ax.get_ylim()[1] for ax in axes[1, :len(datasets)])
    for ax in axes[0, :len(datasets)]:
        ax.set_ylim(0, y_skip_max)
    for ax in axes[1, :len(datasets)]:
        ax.set_ylim(0, y_noskip_max)

    # Row labels
    fig.text(0.01, 0.74, 'Skip-connections', va='center', ha='left', rotation='vertical', fontsize=19, fontweight='bold')
    fig.text(0.01, 0.28, 'No skip-connections', va='center', ha='left', rotation='vertical', fontsize=19, fontweight='bold')

    plt.tight_layout(rect=[0.025, 0, 1, 0.99], pad=2.0)
    return fig, axes


def compute_roc(values_id, values_ood):
    """Compute ROC curve, AUC, and best threshold."""
    y_true = np.concatenate([np.zeros(len(values_id)), np.ones(len(values_ood))])
    y_scores = np.concatenate([values_id, values_ood])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    best_thresh = thresholds[np.argmax(tpr - fpr)]
    return fpr, tpr, auc_score, best_thresh


def plot_roc_curves(datasets, id_skip, ood_skip, id_noskip, ood_noskip, title):
    """Plot ROC curves for skip and no skip datasets."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # SKIP
    values_id = np.concatenate([id_skip[ds] for ds in datasets])
    values_ood = np.concatenate([ood_skip[ds] for ds in datasets])
    fpr, tpr, auc_score, best_thresh = compute_roc(values_id, values_ood)
    axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    axs[0].plot([0, 1], [0, 1], 'k--')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].set_title('Skip-connections')
    axs[0].legend(loc="lower right")
    print(f"Best threshold (Skip): {best_thresh:.6f}")

    # NO SKIP
    values_id = np.concatenate([id_noskip[ds] for ds in datasets])
    values_ood = np.concatenate([ood_noskip[ds] for ds in datasets])
    fpr, tpr, auc_score, best_thresh = compute_roc(values_id, values_ood)
    axs[1].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('No skip-connections')
    axs[1].legend(loc="lower right")
    print(f"Best threshold (No Skip): {best_thresh:.6f}")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig, axs