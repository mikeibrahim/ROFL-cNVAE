#!/usr/bin/env python
"""
Analysis script for the trained VAE model.
Runs full analysis pipeline on the trained model checkpoint.
"""
import os
import sys
import torch
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vae.vae2d import VAE
from vae.config_vae import ConfigVAE
from base.dataset import ROFLDS
from analysis.linear import regress, mi_analysis
from utils.generic import now


def load_model(checkpoint_path, device='cuda:0'):
    """Load trained VAE model from checkpoint."""
    print(f"\n[ANALYSIS] Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model config from checkpoint metadata or reconstruct from checkpoint state
    # For now, we'll create a default config matching the trained model
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
    
    # Create model and load state
    vae = VAE(vae_cfg).to(device).eval()
    vae.load_state_dict(checkpoint['model'])
    
    print(f"[ANALYSIS] Model loaded successfully")
    print(f"[ANALYSIS] Total parameters: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M")
    
    return vae, vae_cfg


def extract_latents(vae, dataset, device='cuda:0', batch_size=300):
    """Extract latent representations for all samples."""
    print(f"\n[ANALYSIS] Extracting latent representations...")
    
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
            
            # Concatenate all latent dimensions
            z_flat = torch.cat([z_i.view(z_i.shape[0], -1) for z_i in z], dim=1)
            latents_list.append(z_flat.cpu().numpy())
    
    latents = np.concatenate(latents_list, axis=0)
    print(f"[ANALYSIS] Extracted latents shape: {latents.shape}")
    
    return latents


def run_analysis(checkpoint_path, device='cuda:0'):
    """Run full analysis pipeline."""
    print(f"\n{'='*80}")
    print(f"[ANALYSIS] Starting VAE Analysis Pipeline ({now(True)})")
    print(f"{'='*80}")
    
    # Load model
    vae, vae_cfg = load_model(checkpoint_path, device)
    
    # Load datasets
    print(f"\n[ANALYSIS] Loading datasets...")
    ds_vld = ROFLDS(vae_cfg.sim_path, 'vld', device=None)
    ds_tst = ROFLDS(vae_cfg.sim_path, 'tst', device=None)
    
    print(f"[ANALYSIS] Validation set: {len(ds_vld)} samples")
    print(f"[ANALYSIS] Test set: {len(ds_tst)} samples")
    
    # Extract latents
    z_vld = extract_latents(vae, ds_vld, device)
    z_tst = extract_latents(vae, ds_tst, device)
    
    # Run regression analysis
    print(f"\n[ANALYSIS] Running linear regression analysis...")
    regr = regress(
        z=z_vld,
        g=ds_vld.g,
        z_tst=z_tst,
        g_tst=ds_tst.g,
    )
    
    # Run auxiliary regression
    regr_aux = regress(
        z=z_vld,
        g=ds_vld.g_aux,
        z_tst=z_tst,
        g_tst=ds_tst.g_aux,
    )
    
    print(f"[ANALYSIS] Main factors R² (mean): {np.nanmean(regr['r2']) * 100:.2f}%")
    print(f"[ANALYSIS] Main factors disentanglement: {regr['d']:.4f}")
    print(f"[ANALYSIS] Main factors completeness: {regr['c']:.4f}")
    print(f"[ANALYSIS] Aux factors R² (mean): {np.nanmean(regr_aux['r2']) * 100:.2f}%")
    
    # Run MI analysis if available
    print(f"\n[ANALYSIS] Running mutual information analysis...")
    mi = mi_analysis(
        z=z_vld,
        g=ds_vld.g,
        n_jobs=int(os.cpu_count() / 2),
    )
    
    print(f"[ANALYSIS] MI (mean): {np.max(mi['mi'], axis=1).mean():.4f}")
    print(f"[ANALYSIS] MI normalized (mean): {np.max(mi['mi_norm'], axis=1).mean():.4f}")
    print(f"[ANALYSIS] MIG (mean): {mi['mig'].mean():.4f}")
    
    # Prepare results
    results = {
        'timestamp': now(True),
        'checkpoint': str(checkpoint_path),
        'model_config': {
            'n_latent_per_group': vae_cfg.n_latent_per_group,
            'n_groups_per_scale': vae_cfg.n_groups_per_scale,
            'n_latent_scales': vae_cfg.n_latent_scales,
            'total_latents': vae.total_latents(),
        },
        'regression_main': {
            'r2_mean': float(np.nanmean(regr['r2'])),
            'r2_std': float(np.nanstd(regr['r2'])),
            'disentanglement': float(regr['d']),
            'completeness': float(regr['c']),
        },
        'regression_aux': {
            'r2_mean': float(np.nanmean(regr_aux['r2'])),
            'r2_std': float(np.nanstd(regr_aux['r2'])),
            'disentanglement': float(regr_aux['d']),
            'completeness': float(regr_aux['c']),
        },
        'mi_analysis': {
            'mi_mean': float(np.max(mi['mi'], axis=1).mean()),
            'mi_norm_mean': float(np.max(mi['mi_norm'], axis=1).mean()),
            'mig_mean': float(mi['mig'].mean()),
        },
    }
    
    # Save results
    results_dir = Path(checkpoint_path).parent.parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    model_name = Path(checkpoint_path).parent.name
    results_file = results_dir / f"{model_name}_analysis_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[ANALYSIS] Results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print(f"[ANALYSIS] Analysis Complete ({now(True)})")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    # Find the latest checkpoint
    models_dir = Path('/home/michael/code/ROFL-cNVAE-fork/models')
    
    # Look for the trained model directory
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
                # Use the final (epoch 160) checkpoint
                final_checkpoint = [c for c in checkpoints if '0160' in c.name]
                if final_checkpoint:
                    checkpoint_path = final_checkpoint[-1]
                else:
                    checkpoint_path = checkpoints[-1]
                
                print(f"\nUsing checkpoint: {checkpoint_path}")
                results = run_analysis(str(checkpoint_path), device='cuda:0')
                break
    else:
        print("ERROR: No suitable checkpoints found!")
        sys.exit(1)
