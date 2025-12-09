#!/usr/bin/env python
"""
Generate mutual information heatmap figure matching Figure 3 from the paper.
Shows MI between latent variables (x-axis) and ground truth factors (y-axis).
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vae.vae2d import VAE
from vae.config_vae import ConfigVAE
from base.dataset import ROFLDS
from analysis.linear import mi_analysis
from utils.generic import now


def load_vae_model(checkpoint_path, device='cuda:0'):
    """Load trained VAE model from checkpoint."""
    print(f"[MI-VIZ] Loading VAE model from checkpoint...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    vae_cfg = ConfigVAE(
        sim='fixate1',
        seed=0,
        n_ch=32,
        res_eps=0.1,
        input_sz=17,
        n_enc_cells=2,
        n_enc_nodes=2,
        n_dec_cells=2,
        n_dec_nodes=1,
        n_pre_cells=3,
        n_pre_blocks=1,
        n_post_cells=3,
        n_post_blocks=1,
        n_latent_scales=3,
        n_latent_per_group=20,
        n_groups_per_scale=20,
        activation_fn='swish',
        weight_norm=True,
        spectral_norm=0,
        ada_groups=True,
        compress=True,
        use_bn=False,
        use_se=True,
        balanced_recon=True,
        residual_kl=True,
        scale_init=False,
        separable=False,
        save=False,
    )
    
    vae = VAE(vae_cfg).to(device).eval()
    vae.load_state_dict(checkpoint['model'])
    
    print(f"[MI-VIZ] Model loaded successfully ({sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params)")
    
    return vae, vae_cfg


def extract_latents_batch(vae, dataset, device='cuda:0', batch_size=300):
    """Extract latent representations for all samples."""
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    latents_list = []
    with torch.no_grad():
        for x, _ in dataloader:
            if x.device != device:
                x = x.to(device)
            
            z, _, _, q, p = vae.xtract_ftr(x=x, t=0.0)
            z_flat = torch.cat([z_i.view(z_i.shape[0], -1) for z_i in z], dim=1)
            latents_list.append(z_flat.cpu().numpy())
    
    return np.concatenate(latents_list, axis=0)


def create_mi_heatmap(checkpoint_path, device='cuda:0'):
    """Generate MI heatmap figure matching Figure 3."""
    print(f"\n{'='*80}")
    print(f"[MI-VIZ] Generating Mutual Information Heatmap ({now(True)})")
    print(f"{'='*80}")
    
    # Load model
    vae, vae_cfg = load_vae_model(checkpoint_path, device)
    
    # Load datasets
    print(f"\n[MI-VIZ] Loading dataset...")
    ds_vld = ROFLDS(vae_cfg.sim_path, 'vld', device=None)
    
    print(f"[MI-VIZ] Validation set: {len(ds_vld)} samples")
    
    # Extract latents
    print(f"[MI-VIZ] Extracting latent representations...")
    z_vld = extract_latents_batch(vae, ds_vld, device)
    print(f"[MI-VIZ] Latent shape: {z_vld.shape}")
    
    # Compute MI
    print(f"\n[MI-VIZ] Computing mutual information (this may take a few minutes)...")
    mi_result = mi_analysis(
        z=z_vld,
        g=ds_vld.g,
        n_jobs=int(os.cpu_count() / 2),
    )
    
    # Extract MI matrix
    mi_matrix = mi_result['mi']  # Shape: (n_latents, n_factors)
    print(f"[MI-VIZ] MI matrix shape: {mi_matrix.shape}")
    
    # Create figure with proper layout (larger to accommodate labels)
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Get number of latents and factors
    n_latents = mi_matrix.shape[0]
    n_factors = mi_matrix.shape[1]
    
    # Create heatmap
    im = ax.imshow(mi_matrix.T, cmap='Blues', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_xlabel('Latent Variables (x-axis)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Ground Truth Factors (y-axis)', fontsize=16, fontweight='bold')
    ax.set_title('Figure 3: Mutual Information between Latent Variables and Ground Truth Factors\nVAE (Hierarchical - 700-dim latent space)', 
                 fontsize=16, fontweight='bold', pad=25)
    
    # Set x-axis ticks (every 50 latents for clarity)
    x_ticks = np.arange(0, n_latents, 50)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(i)}' for i in x_ticks], fontsize=12)
    
    # Set y-axis ticks more sparsely (every 50 factors to avoid crowding)
    y_ticks = np.arange(0, n_factors, 50)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'F{int(i)}' for i in y_ticks], fontsize=12)
    
    # Add colorbar with better sizing
    cbar = plt.colorbar(im, ax=ax, label='Mutual Information (bits)', pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    
    # Add grid for readability (every 50 units)
    ax.set_xticks(np.arange(0, n_latents, 50), minor=False)
    ax.set_yticks(np.arange(0, n_factors, 50), minor=False)
    
    # Add minor grid for finer detail
    ax.set_xticks(np.arange(-0.5, n_latents, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, n_factors, 10), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.2)
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add hierarchical scale indicators (similar to paper)
    # Mark scale boundaries with red dashed lines
    scale_boundaries = [20, 360, 700]  # Boundaries between scales
    for boundary in scale_boundaries[:-1]:
        ax.axvline(x=boundary-0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add scale labels at top
    scale_labels = [
        {'x': 10, 'label': '2×2\nScale', 'width': 20},
        {'x': 190, 'label': '4×4\nScale', 'width': 340},
        {'x': 530, 'label': '8×8\nScale', 'width': 340}
    ]
    
    for scale in scale_labels:
        ax.text(scale['x'], -50, scale['label'], fontsize=11, fontweight='bold',
               ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        # Draw bracket
        ax.plot([scale['x']-scale['width']/2, scale['x']+scale['width']/2], 
               [-30, -30], 'k-', linewidth=2)
    
    # Add text box with statistics
    stats_text = f"MI Statistics:\nMax: {np.max(mi_matrix):.4f} bits\nMean: {np.mean(mi_matrix):.4f} bits\nStd: {np.std(mi_matrix):.4f} bits"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(checkpoint_path).parent.parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    fig_path = output_dir / 'mi_heatmap_vae.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"[MI-VIZ] Figure saved to: {fig_path}")
    
    # Also save as PDF for publication
    fig_path_pdf = output_dir / 'mi_heatmap_vae.pdf'
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"[MI-VIZ] PDF saved to: {fig_path_pdf}")
    
    plt.close()
    
    # Save MI matrix as numpy file
    mi_file = output_dir / 'mi_matrix.npy'
    np.save(mi_file, mi_matrix)
    print(f"[MI-VIZ] MI matrix saved to: {mi_file}")
    
    # Compute and display statistics
    print(f"\n[MI-VIZ] MI Statistics:")
    print(f"  Max MI: {np.max(mi_matrix):.4f} bits")
    print(f"  Mean MI: {np.mean(mi_matrix):.4f} bits")
    print(f"  Min MI: {np.min(mi_matrix):.4f} bits")
    print(f"  Std MI: {np.std(mi_matrix):.4f} bits")
    
    print(f"\n[MI-VIZ] Per-factor max MI:")
    for i, factor_name in enumerate(factor_names):
        max_mi = np.max(mi_matrix[:, i])
        best_latent = np.argmax(mi_matrix[:, i])
        print(f"  {factor_name:15s}: {max_mi:.4f} (latent {best_latent})")
    
    print(f"\n{'='*80}")
    print(f"[MI-VIZ] Heatmap Generation Complete ({now(True)})")
    print(f"{'='*80}\n")
    
    return mi_matrix, factor_names


if __name__ == "__main__":
    # Find the latest checkpoint
    models_dir = Path('/home/michael/code/ROFL-cNVAE-fork/models')
    
    model_dirs = sorted([d for d in models_dir.glob('fixate1*') if d.is_dir()])
    
    if not model_dirs:
        print("ERROR: No trained models found!")
        sys.exit(1)
    
    # Get the most recent training run
    for model_dir in reversed(model_dirs):
        run_dirs = sorted([d for d in model_dir.glob('ep160*') if d.is_dir()])
        if run_dirs:
            latest_run = run_dirs[-1]
            checkpoints = sorted(latest_run.glob('VAE+*.pt'))
            if checkpoints:
                final_checkpoint = [c for c in checkpoints if '0160' in c.name]
                if final_checkpoint:
                    checkpoint_path = final_checkpoint[-1]
                else:
                    checkpoint_path = checkpoints[-1]
                
                print(f"\nUsing checkpoint: {checkpoint_path}\n")
                mi_matrix, factor_names = create_mi_heatmap(str(checkpoint_path), device='cuda:0')
                break
    else:
        print("ERROR: No suitable checkpoints found!")
        sys.exit(1)
