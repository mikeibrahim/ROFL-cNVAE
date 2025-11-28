#!/usr/bin/env python3
"""
Simplified training script for cNVAE model.
Works around PyTorch compatibility issues with the standard trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from vae.vae2d import VAE
from vae.config_vae import ConfigVAE, ConfigTrainVAE
from base.dataset import ROFLDS


class SimpleTrainer:
    """Simplified trainer for cNVAE model"""
    
    def __init__(self, model_config, train_config, base_dir='/Users/mike/berkeley/rctn/ROFL-cNVAE'):
        self.base_dir = Path(base_dir)
        self.device = torch.device('cpu')
        
        # Setup model
        self.cfg_vae = ConfigVAE(**model_config)
        self.model = VAE(self.cfg_vae).to(self.device)
        
        # Setup training config
        self.cfg_train = ConfigTrainVAE(**train_config)
        
        # Setup data
        self.setup_data()
        
        # Setup optimizer
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.cfg_train.lr)
        
        print(f"✓ Trainer initialized")
        print(f"  Model: {self.cfg_vae.name()}")
        print(f"  Device: {self.device}")
        print(f"  Training samples: {len(self.dl_trn) * self.cfg_train.batch_size}")
        
    def setup_data(self):
        """Setup data loaders"""
        sim_path = self.base_dir / 'data' / 'fixate1_dim-17_n-750k'
        
        ds_trn = ROFLDS(str(sim_path), 'trn', device=None)
        ds_vld = ROFLDS(str(sim_path), 'vld', device=None)
        ds_tst = ROFLDS(str(sim_path), 'tst', device=None)
        
        self.dl_trn = DataLoader(
            ds_trn, batch_size=self.cfg_train.batch_size,
            shuffle=True, drop_last=True, num_workers=0
        )
        self.dl_vld = DataLoader(
            ds_vld, batch_size=self.cfg_train.batch_size,
            shuffle=False, drop_last=False, num_workers=0
        )
        self.dl_tst = DataLoader(
            ds_tst, batch_size=self.cfg_train.batch_size,
            shuffle=False, drop_last=False, num_workers=0
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.dl_trn, desc=f'Epoch {epoch+1}')
        for batch_idx, (x, norm) in enumerate(pbar):
            x = x.to(self.device)
            norm = norm.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            y, _, q, p = self.model(x)
            
            # Compute loss components
            epe = self.model.loss_recon(x=x, y=y, w=1/norm)
            kl_all, kl_diag = self.model.loss_kl(q, p)
            
            # Total loss
            loss = torch.mean(epe) + torch.mean(sum(kl_all))
            
            # Backward pass
            loss.backward()
            if self.cfg_train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg_train.grad_clip
                )
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.dl_trn)
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, norm in self.dl_vld:
                x = x.to(self.device)
                norm = norm.to(self.device)
                
                y, _, q, p = self.model(x)
                epe = self.model.loss_recon(x=x, y=y, w=1/norm)
                kl_all, kl_diag = self.model.loss_kl(q, p)
                loss = torch.mean(epe) + torch.mean(sum(kl_all))
                total_loss += loss.item()
        
        return total_loss / len(self.dl_vld)
    
    def train(self, epochs=None, save_dir=None):
        """Train the model"""
        epochs = epochs or self.cfg_train.epochs
        save_dir = Path(save_dir) if save_dir else self.base_dir / 'models' / self.cfg_vae.name()
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Save directory: {save_dir}")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, save_dir / f'checkpoint_best.pt')
                print(f"  ✓ Checkpoint saved")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.cfg_train.chkpt_freq == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, save_dir / f'checkpoint_{epoch+1:04d}.pt')
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config_model': self.cfg_vae.__dict__ if hasattr(self.cfg_vae, '__dict__') else {},
            'config_train': self.cfg_train.__dict__ if hasattr(self.cfg_train, '__dict__') else {},
        }
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump({k: v for k, v in history.items() if isinstance(v, (list, dict, int, float, str))}, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    import sys
    
    # Model configuration
    model_config = {
        'sim': 'fixate1',
        'n_ch': 32,
        'input_sz': 17,
        'ker_sz': 2,
        'res_eps': 0.1,
        'n_enc_cells': 2,
        'n_enc_nodes': 2,
        'n_dec_cells': 2,
        'n_dec_nodes': 1,
        'n_pre_cells': 3,
        'n_pre_blocks': 1,
        'n_post_cells': 3,
        'n_post_blocks': 1,
        'n_latent_scales': 3,
        'n_latent_per_group': 16,
        'n_groups_per_scale': 20,
        'activation_fn': 'swish',
        'balanced_recon': True,
        'residual_kl': True,
        'ada_groups': True,
        'compress': True,
        'use_se': True,
        'use_bn': False,
        'weight_norm': False,
        'spectral_norm': 0,
        'full': False,
        'save': False,
        'base_dir': '/Users/mike/berkeley/rctn/ROFL-cNVAE',
        'seed': 0,
    }
    
    # Training configuration
    train_config = {
        'lr': 0.002,
        'epochs': 160,
        'batch_size': 600,
        'warm_restart': 0,
        'warmup_portion': 0.0125,
        'optimizer': 'adamax_fast',
        'scheduler_type': 'cosine',
        'ema_rate': 0.999,
        'grad_clip': 250,
        'use_amp': False,
        'kl_beta': 1.0,
        'kl_beta_min': 1e-4,
        'kl_balancer': 'equal',
        'kl_anneal_cycles': 0,
        'kl_anneal_portion': 0.5,
        'kl_const_portion': 0.01,
        'lambda_anneal': True,
        'lambda_norm': 1e-3,
        'lambda_init': 1e-7,
        'spectral_reg': False,
        'chkpt_freq': 10,
        'eval_freq': 2,
        'log_freq': 10,
    }
    
    # Parse command line arguments
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    # Create trainer and train
    trainer = SimpleTrainer(model_config, train_config)
    trainer.train(epochs=epochs)
