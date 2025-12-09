"""
Multi-GPU training script for VAE using DistributedDataParallel (DDP).
Uses all 4 NVIDIA GPUs for parallel training.
"""
import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vae.vae2d import VAE
from vae.config_vae import ConfigVAE, ConfigTrainVAE
from vae.train_vae import TrainerVAE
from base.dataset import ROFLDS
from base.utils_model import *


class TrainerVAE_DDP(TrainerVAE):
    """Extended TrainerVAE with DistributedDataParallel support."""
    
    def __init__(
            self,
            model: VAE,
            cfg: ConfigTrainVAE,
            rank: int,
            world_size: int,
            ema: bool = True,
            **kwargs,
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self._base_model = model  # Store reference before DDP wrapping
        
        # Override device based on rank
        kwargs['device'] = f'cuda:{rank}'
        
        # Call parent init (this calls setup_data before DDP wrapping)
        super(TrainerVAE_DDP, self).__init__(
            model=model, cfg=cfg, ema=ema, **kwargs)
        
        # Wrap model with DDP after parent init
        self.model = DDP(self.model, device_ids=[rank])
        if self.model_ema is not None:
            # EMA model doesn't need DDP, only used for evaluation
            pass

    @property
    def unwrapped_model(self):
        """Get the unwrapped model (without DDP wrapper)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    @property
    def model_cfg(self):
        """Get model config for DDP compatibility."""
        return self.unwrapped_model.cfg

    def _get_model_cfg(self):
        """Get model config, handling both DDP wrapped and unwrapped models."""
        return self.unwrapped_model.cfg

    def setup_data(self, gpu: bool = True):
        """Setup distributed data loaders."""
        device = self.device if gpu else None
        # Use _base_model or unwrapped model for config access
        model_cfg = self._base_model.cfg if hasattr(self, '_base_model') else self._get_model_cfg()
        ds_trn = ROFLDS(model_cfg.sim_path, 'trn', device)
        ds_vld = ROFLDS(model_cfg.sim_path, 'vld', device)
        ds_tst = ROFLDS(model_cfg.sim_path, 'tst', device)
        
        # Use DistributedSampler for training
        self.train_sampler = DistributedSampler(
            ds_trn,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.shuffle,
        )
        
        # Create dataloaders
        # When data is on GPU, don't use multiprocessing workers
        kws = dict(
            batch_size=self.cfg.batch_size // self.world_size,  # Divide batch across GPUs
            drop_last=True,
            num_workers=0,  # Data is already on GPU, no need for workers
            pin_memory=False,
        )
        self.dl_trn = DataLoader(ds_trn, sampler=self.train_sampler, **kws)
        
        kws.update({'drop_last': False})
        # Validation and test don't need distributed sampler
        self.dl_vld = DataLoader(ds_vld, shuffle=False, **kws)
        self.dl_tst = DataLoader(ds_tst, shuffle=False, **kws)
        return

    def select_model(self, ema: bool = False):
        """Override to return unwrapped model for validation/testing."""
        if ema:
            assert self.model_ema is not None
            return self.model_ema.eval()
        # Return the unwrapped model for inference methods like xtract_ftr
        return self.unwrapped_model.eval()

    def train(
            self,
            comment: str = None,
            epochs: Union[int, range] = None,
            save: bool = True, ):
        epochs = epochs if epochs else self.cfg.epochs
        assert isinstance(epochs, (int, range)), "allowed: {int, range}"
        epochs = range(epochs) if isinstance(epochs, int) else epochs
        comment = comment if comment else self.cfg.name()
        kwargs = dict(n_iters_warmup=int(np.round(
            self.n_iters * self.cfg.warmup_portion)))
        self.stats.clear()
        
        if save and self.is_main:
            self.unwrapped_model.create_chkpt_dir(comment)
            self.cfg.save(self.unwrapped_model.chkpt_dir)
            writer = pjoin(
                self.unwrapped_model.cfg.runs_dir,
                os.path.basename(self.unwrapped_model.chkpt_dir),
            )
            self.writer = SummaryWriter(writer)
            self.logger = make_logger(
                name=type(self).__name__,
                path=self.unwrapped_model.chkpt_dir,
                level=logging.WARNING,
            )
        
        if self.cfg.scheduler_type == 'cosine':
            self.optim_schedule.T_max *= len(self.dl_trn)
        else:
            raise NotImplementedError

        if self.is_main:
            self.pbar = tqdm(
                epochs,
                dynamic_ncols=True,
                leave=True,
                position=0,
            )
        else:
            self.pbar = epochs
            
        for epoch in self.pbar:
            # Set epoch for distributed sampler
            self.train_sampler.set_epoch(epoch)
            
            avg_loss = self.iteration(epoch, **kwargs)
            
            if self.is_main:
                msg = ', '.join([
                    f"epoch # {epoch + 1:d}",
                    f"avg loss: {avg_loss:3f}",
                ])
                self.pbar.set_description(msg)
                
            if not save:
                continue
            if self.is_main and (epoch + 1) % self.cfg.chkpt_freq == 0:
                self.save(
                    checkpoint=epoch + 1,
                    path=self.unwrapped_model.chkpt_dir,
                )
            if self.is_main and (epoch + 1) % self.cfg.eval_freq == 0:
                gstep = (epoch + 1) * len(self.dl_trn)
                _ = self.validate(gstep)
                
        if self.writer is not None:
            self.writer.close()
        return

    def iteration(self, epoch: int = 0, **kwargs):
        self.model.train()
        nelbo = AvgrageMeter()
        grads = AvgrageMeter()
        perdim_kl = AvgrageMeter()
        perdim_epe = AvgrageMeter()
        
        for i, (x, norm) in enumerate(self.dl_trn):
            gstep = epoch * len(self.dl_trn) + i
            # warm-up lr
            if gstep < kwargs['n_iters_warmup']:
                lr = self.cfg.lr * gstep / kwargs['n_iters_warmup']
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr
            # send to device
            if x.device != self.device:
                x, norm = self.to([x, norm])
            # zero grad
            self.optim.zero_grad(set_to_none=True)
            # forward + loss
            with torch.cuda.amp.autocast(enabled=self.cfg.use_amp):
                y, _, q, p = self.model(x)
                epe = self.unwrapped_model.loss_recon(x=x, y=y, w=1/norm)
                kl_all, kl_diag = self.unwrapped_model.loss_kl(q, p)
                # balance kl
                balanced_kl, gamma, kl_vals = kl_balancer(
                    kl_all=kl_all,
                    alpha=self.alphas,
                    coeff=self.betas[gstep] if gstep < len(self.betas) else self.betas[-1],
                    beta=self.cfg.kl_beta,
                )
                loss = torch.mean(epe + balanced_kl)
                # add regularization
                loss_w = self.unwrapped_model.loss_weight()
                wd_coeff = self.wd_coeffs[gstep] if gstep < len(self.wd_coeffs) else self.wd_coeffs[-1]
                if wd_coeff > 0 and loss_w is not None:
                    loss += wd_coeff * loss_w
                cond_reg_spectral = self.cfg.lambda_norm > 0 \
                    and self.cfg.spectral_reg and \
                    not self.unwrapped_model.cfg.spectral_norm
                if cond_reg_spectral:
                    loss_sr = self.unwrapped_model.loss_spectral(
                        device=self.device, name='w')
                    loss += wd_coeff * loss_sr
                else:
                    loss_sr = None
            # backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optim)
            # clip grad
            if self.cfg.grad_clip is not None:
                if gstep < kwargs['n_iters_warmup']:
                    max_norm = self.cfg.grad_clip * 2
                else:
                    max_norm = self.cfg.grad_clip
                grad_norm = nn.utils.clip_grad_norm_(
                    parameters=self.parameters(),
                    max_norm=max_norm,
                ).item()
                grads.update(grad_norm)
                if self.is_main:
                    self.stats['grad'][gstep] = grad_norm
                if grad_norm > self.cfg.grad_clip:
                    if self.is_main:
                        self.stats['loss'][gstep] = loss.item()
            # update average meters & stats
            nelbo.update(loss.item())
            perdim_kl.update(torch.stack(kl_diag).mean().item())
            perdim_epe.update(epe.mean().item() / self.unwrapped_model.cfg.input_sz ** 2)
            
            if self.is_main:
                msg = [
                    f"gstep # {gstep:.3g}",
                    f"nelbo: {nelbo.avg:0.3f}",
                ]
                if self.cfg.grad_clip:
                    msg += [f"grad: {grads.val:0.1f}"]
                self.pbar.set_description(', '.join(msg))
            # step
            self.scaler.step(self.optim)
            self.scaler.update()
            self.update_ema()
            # optim schedule
            cond_schedule = (
                gstep > kwargs['n_iters_warmup']
                and self.optim_schedule is not None
            )
            if cond_schedule:
                self.optim_schedule.step()
            # write
            cond_write = (
                self.is_main and
                gstep > 0 and
                self.writer is not None and
                gstep % self.cfg.log_freq == 0
            )
            if not cond_write:
                continue
            beta_idx = min(gstep, len(self.betas) - 1)
            wd_idx = min(gstep, len(self.wd_coeffs) - 1)
            to_write = {
                'train/beta': self.betas[beta_idx],
                'train/reg_coeff': self.wd_coeffs[wd_idx],
                'train/lr': self.optim.param_groups[0]['lr'],
                'train/loss_kl': torch.mean(sum(kl_all)).item(),
                'train/loss_epe': torch.mean(epe).item(),
                'train/nelbo_avg': nelbo.avg,
                'train/perdim_kl': perdim_kl.avg,
                'train/perdim_epe': perdim_epe.avg,
                'train/reg_weight': 0 if loss_w is None
                else loss_w.item(),
            }
            if self.cfg.grad_clip is not None:
                to_write['train/grad_norm'] = grads.avg
            if cond_reg_spectral:
                to_write['train/reg_spectral'] = loss_sr.item()
            total_active = 0
            for j, kl_diag_i in enumerate(kl_diag):
                to_write[f"kl_full/gamma_layer_{j}"] = gamma[j].item()
                to_write[f"kl_full/vals_layer_{j}"] = kl_vals[j].item()
                n_active = torch.sum(kl_diag_i > 0.1).item()
                to_write[f"kl_full/active_{j}"] = n_active
                total_active += n_active
            to_write['train/total_active'] = total_active
            ratio = total_active / self.unwrapped_model.total_latents()
            to_write['train/total_active_ratio'] = ratio
            for k, v in to_write.items():
                self.writer.add_scalar(k, v, gstep)
            # reset average meters
            if gstep % (self.cfg.log_freq * 10) == 0:
                grads.reset()
                nelbo.reset()

        return nelbo.avg

    def parameters(self, requires_grad: bool = True):
        if requires_grad:
            return filter(
                lambda p: p.requires_grad,
                self.unwrapped_model.parameters(),
            )
        else:
            return self.unwrapped_model.parameters()

    def save(self, path: str, checkpoint: int = None):
        if not self.is_main:
            return
        if checkpoint is not None:
            global_step = checkpoint * len(self.dl_trn)
        else:
            global_step = None
        state_dict = {
            'metadata': {
                'checkpoint': checkpoint,
                'global_step': global_step,
                'stats': self.stats},
            'model': self.unwrapped_model.state_dict(),
            'model_ema': self.model_ema.state_dict()
            if self.model_ema is not None else None,
            'optim': self.optim.state_dict(),
            'scaler': self.scaler.state_dict(),
            'scheduler': self.optim_schedule.state_dict()
            if self.optim_schedule is not None else None,
        }
        fname = '+'.join([
            type(self.unwrapped_model).__name__,
            type(self).__name__],
        )
        if checkpoint is not None:
            fname = '-'.join([
                fname,
                f"{checkpoint:04d}"
            ])
        fname = f"{fname}_({now(True)}).pt"
        fname = pjoin(path, fname)
        torch.save(state_dict, fname)
        return

    def reset_model(self):
        self.model = VAE(self.unwrapped_model.cfg).to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])
        self.model_ema = VAE(self.unwrapped_model.cfg).to(self.device)
        return

    def plot(self, sample: dict = None, regr: dict = None, **kwargs):
        """Override plot to use unwrapped_model for DDP compatibility."""
        from analysis.linear import mi_analysis
        from figures.fighelper import plot_heatmap, show_opticflow, plot_bar
        
        regr = regr if regr else self.regress(
            **filter_kwargs(self.regress, kwargs))
        if sample is None:
            x_sample, z_sample = self.sample(
                **filter_kwargs(self.sample, kwargs))
        else:
            x_sample, z_sample = sample['x'], sample['z']

        figs = {}
        # samples (opticflow)
        fig, _ = show_opticflow(
            x_sample, n=6, display=False)
        figs['fig/sample'] = fig

        # corr (regression)
        f = self.dl_tst.dataset.f
        _tx = [f"({i:02d})" for i in range(len(f))]
        _ty = [f"{e} ({i:02d})" for i, e in enumerate(f)]
        rd = np.diag(regr['regr/r'])
        title = f"all  =  {rd.mean():0.3f} ± {rd.std():0.3f}  "
        title += r'$(\mu \pm \sigma)$' + '\n'
        name_groups = collections.defaultdict(list)
        for i, lbl in enumerate(f):
            k = lbl.split('_')[0]
            name_groups[k].append(i)
        for i, (k, ids) in enumerate(name_groups.items()):
            title += f"{k} :  {rd[ids].mean():0.2f},"
            title += ' ' * 5
            if (i + 1) % 3 == 0:
                title += '\n'
        fig, _ = plot_heatmap(
            r=regr['regr/r'],
            title=title,
            cmap='PiYG',
            xticklabels=_tx,
            yticklabels=_ty,
            annot_kws={'fontsize': 12},
            figsize=(0.72 * len(f), 0.6 * len(f)),
            display=False,
        )
        figs['fig/regression'] = fig

        # barplots
        df = pd.DataFrame({
            'x': self.dl_vld.dataset.f,
            'y': regr['regr/r2'],
        })
        fig, _ = plot_bar(df, tick_labelsize_x=10, display=False)
        figs['fig/bar'] = fig
        # aux
        df = pd.DataFrame({
            'x': self.dl_vld.dataset.f_aux,
            'y': regr['regr/aux/r2'],
        })
        fig, _ = plot_bar(df, tick_labelsize_x=10, display=False)
        figs['fig/bar_aux'] = fig

        # Use unwrapped_model instead of self.model for DDP compatibility
        if self.unwrapped_model.cfg.compress:  # only for cNVAE
            n_jobs = max(1, joblib.effective_n_jobs())
            n_jobs /= max(1, torch.cuda.device_count())
            mi = mi_analysis(
                z=regr['z_vld'],
                g=self.dl_vld.dataset.g,
                n_jobs=int(n_jobs),
            )
            mi = {
                f"regr/{k}": v for
                k, v in mi.items()
            }
            regr = {**regr, **mi}
            title = '_'.join(self.unwrapped_model.cfg.name().split('_')[:3])
            mi_max = np.round(np.max(regr['regr/mi'], axis=1), 2)
            mi_max = ', '.join([str(e) for e in mi_max])
            title = f"model = {title};    max MI (row) = {mi_max}"
            figsize = (0.08 * self.unwrapped_model.total_latents(), 0.72 * len(f))
            fig, _ = plot_heatmap(
                r=regr['regr/mi'],
                yticklabels=_ty,
                title=title,
                tick_labelsize_x=10,
                tick_labelsize_y=7,
                title_fontsize=14,
                title_y=1.02,
                vmin=0,
                vmax=0.65,
                cmap='rocket',
                linecolor='dimgrey',
                figsize=figsize,
                cbar=False,
                annot=False,
                display=False,
            )
            figs['fig/mutual_info'] = fig
        return x_sample, z_sample, regr, figs


def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def train_worker(rank, world_size, args):
    """Worker function for each GPU."""
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"\n[DDP] Starting training on {world_size} GPUs")
        print(args)
    
    # Create model
    vae = VAE(ConfigVAE(
        sim=args.sim,
        seed=args.seed,
        n_ch=args.n_ch,
        res_eps=args.res_eps,
        input_sz=args.input_sz,
        n_enc_cells=args.n_enc_cells,
        n_enc_nodes=args.n_enc_nodes,
        n_dec_cells=args.n_dec_cells,
        n_dec_nodes=args.n_dec_nodes,
        n_pre_cells=args.n_pre_cells,
        n_pre_blocks=args.n_pre_blocks,
        n_post_cells=args.n_post_cells,
        n_post_blocks=args.n_post_blocks,
        n_latent_scales=args.n_latent_scales,
        n_latent_per_group=args.n_latent_per_group,
        n_groups_per_scale=args.n_groups_per_scale,
        activation_fn=args.activation_fn,
        spectral_norm=args.spectral_norm,
        weight_norm=args.weight_norm,
        ada_groups=args.ada_groups,
        compress=args.compress,
        save=not args.dry_run and rank == 0,
        use_bn=args.use_bn,
        use_se=args.use_se,
        balanced_recon=True,
        residual_kl=True,
        scale_init=False,
        separable=False,
    ))
    
    # Create trainer with DDP
    tr = TrainerVAE_DDP(
        model=vae,
        rank=rank,
        world_size=world_size,
        cfg=ConfigTrainVAE(
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            warm_restart=args.warm_restart,
            warmup_portion=args.warmup_portion,
            optimizer=args.optimizer,
            grad_clip=args.grad_clip,
            use_amp=args.use_amp,
            # kl
            kl_beta=args.kl_beta,
            kl_balancer=args.kl_balancer,
            kl_anneal_portion=args.kl_anneal_portion,
            kl_const_portion=args.kl_const_portion,
            kl_anneal_cycles=args.kl_anneal_cycles,
            # weight reg
            lambda_anneal=args.lambda_anneal,
            lambda_norm=args.lambda_norm,
            lambda_init=1e-7,
            # freqs
            chkpt_freq=args.chkpt_freq,
            eval_freq=args.eval_freq,
            log_freq=args.log_freq),
    )
    
    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    if args.resume is not None:
        if rank == 0:
            print(f"[DDP] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}', weights_only=False)
        tr.unwrapped_model.load_state_dict(checkpoint['model'])
        if tr.model_ema is not None and checkpoint.get('model_ema') is not None:
            tr.model_ema.load_state_dict(checkpoint['model_ema'])
        tr.optim.load_state_dict(checkpoint['optim'])
        tr.scaler.load_state_dict(checkpoint['scaler'])
        if tr.optim_schedule is not None and checkpoint.get('scheduler') is not None:
            tr.optim_schedule.load_state_dict(checkpoint['scheduler'])
        if checkpoint.get('metadata') and checkpoint['metadata'].get('checkpoint'):
            start_epoch = checkpoint['metadata']['checkpoint']
            if rank == 0:
                print(f"[DDP] Resuming from epoch {start_epoch}")
    
    if rank == 0:
        msg = ', '.join([
            f"# enc ftrs: {sum(vae.ftr_sizes()[0].values())}",
            f"# conv layers: {len(vae.all_conv_layers)}",
            f"# latents: {vae.total_latents()}",
        ])
        print('\n', msg)
        vae.print()
        msg = '\n'.join([
            f"VAE:\t\t{vae.cfg.name()}",
            f"Trainer:\t{tr.cfg.name()}\n",
        ])
        print(msg)

    if args.comment is not None:
        comment = '_'.join([
            args.comment,
            tr.cfg.name(),
        ])
    else:
        comment = tr.cfg.name()

    if not args.dry_run:
        tr.train(comment, epochs=range(start_epoch, args.epochs))

    if rank == 0:
        print(f"\n[PROGRESS] fitting VAE on {world_size} GPUs done ({now(True)}).\n")
    
    cleanup_ddp()


def true_fn(v):
    return v.lower() in ('yes', 'true', 't', '1')


def _setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sim",
        help='simulation category',
        type=str,
    )
    parser.add_argument(
        "--n_gpus",
        help='number of GPUs to use',
        default=4,
        type=int,
    )
    parser.add_argument(
        "--comment",
        help='comment',
        default=None,
        type=str,
    )
    parser.add_argument(
        "--n_ch",
        help='# channels',
        default=32,
        type=int,
    )
    parser.add_argument(
        "--input_sz",
        help='ROFL dim',
        default=17,
        type=int,
    )
    parser.add_argument(
        "--res_eps",
        help='x + eps * f(x)',
        default=0.1,
        type=float,
    )
    # enc
    parser.add_argument(
        "--n_enc_cells",
        help='# enc cells',
        default=2,
        type=int,
    )
    parser.add_argument(
        "--n_enc_nodes",
        help='# enc nodes',
        default=2,
        type=int,
    )
    # dec
    parser.add_argument(
        "--n_dec_cells",
        help='# dec cells',
        default=2,
        type=int,
    )
    parser.add_argument(
        "--n_dec_nodes",
        help='# dec nodes',
        default=1,
        type=int,
    )
    # pre
    parser.add_argument(
        "--n_pre_cells",
        help='# preprocessing cells',
        default=3,
        type=int,
    )
    parser.add_argument(
        "--n_pre_blocks",
        help='# preprocessing blocks',
        default=1,
        type=int,
    )
    # post
    parser.add_argument(
        "--n_post_cells",
        help='# postprocessing cells',
        default=3,
        type=int,
    )
    parser.add_argument(
        "--n_post_blocks",
        help='# postprocessing blocks',
        default=1,
        type=int,
    )
    # latents
    parser.add_argument(
        "--n_latent_scales",
        help='# latent scales',
        default=3,
        type=int,
    )
    parser.add_argument(
        "--n_latent_per_group",
        help='# latents per group',
        default=20,
        type=int,
    )
    parser.add_argument(
        "--n_groups_per_scale",
        help='# groups per scale',
        default=20,
        type=int,
    )
    parser.add_argument(
        "--activation_fn",
        help='activation function',
        default='swish',
        type=str,
    )
    parser.add_argument(
        "--weight_norm",
        help='weight norm (disable to use soft reg)',
        default=True,
        type=true_fn,
    )
    parser.add_argument(
        "--spectral_norm",
        help='spectral norm (0 = disable)',
        default=0,
        type=int,
    )
    parser.add_argument(
        "--ada_groups",
        help='adaptive latent groups?',
        default=True,
        type=true_fn,
    )
    parser.add_argument(
        "--compress",
        help='compress latent space?',
        default=True,
        type=true_fn,
    )
    parser.add_argument(
        "--use_bn",
        help='use batch norm?',
        default=False,
        type=true_fn,
    )
    parser.add_argument(
        "--use_se",
        help='use squeeze & excite?',
        default=True,
        type=true_fn,
    )
    # training
    parser.add_argument(
        "--lr",
        help='learning rate',
        default=0.002,
        type=float,
    )
    parser.add_argument(
        "--epochs",
        help='# epochs',
        default=160,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help='batch size (total across all GPUs)',
        default=1200,
        type=int,
    )
    parser.add_argument(
        "--warm_restart",
        help='# warm restarts',
        default=0,
        type=int,
    )
    parser.add_argument(
        "--warmup_portion",
        help='warmup portion',
        default=1.25e-2,
        type=float,
    )
    parser.add_argument(
        "--optimizer",
        help='optimizer',
        default='adamax_fast',
        type=str,
    )
    parser.add_argument(
        "--kl_beta",
        help='kl loss beta coefficient',
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--kl_balancer",
        help='kl balancer function',
        default='equal',
        type=str,
    )
    parser.add_argument(
        "--kl_anneal_portion",
        help='kl beta anneal portion',
        default=0.3,
        type=float,
    )
    parser.add_argument(
        "--kl_const_portion",
        help='kl const portion',
        default=1e-2,
        type=float,
    )
    parser.add_argument(
        "--kl_anneal_cycles",
        help='0: linear, >0: cosine',
        default=0,
        type=int,
    )
    parser.add_argument(
        "--lambda_anneal",
        help='anneal weight reg coeff?',
        default=False,
        type=true_fn,
    )
    parser.add_argument(
        "--lambda_norm",
        help='weight regularization strength',
        default=1e-2,
        type=float,
    )
    parser.add_argument(
        "--grad_clip",
        help='gradient norm clipping',
        default=250.0,
        type=float,
    )
    parser.add_argument(
        "--seed",
        help='random seed',
        default=0,
        type=int,
    )
    parser.add_argument(
        "--chkpt_freq",
        help='checkpoint freq',
        default=10,
        type=int,
    )
    parser.add_argument(
        "--eval_freq",
        help='eval freq',
        default=10,
        type=int,
    )
    parser.add_argument(
        "--log_freq",
        help='log freq',
        default=10,
        type=int,
    )
    parser.add_argument(
        "--use_amp",
        help='automatic mixed precision?',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--dry_run",
        help='to make sure config is alright',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--resume",
        help='path to checkpoint to resume from',
        default=None,
        type=str,
    )
    parser.add_argument(
        "--start_epoch",
        help='epoch to start from when resuming',
        default=0,
        type=int,
    )
    return parser.parse_args()


def main():
    args = _setup_args()
    world_size = min(args.n_gpus, torch.cuda.device_count())
    
    print(f"[DDP] Using {world_size} GPUs for training")
    
    mp.spawn(
        train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
