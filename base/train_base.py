from base.utils_model import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(object):
	def __init__(
			self,
			model: Module,
			cfg: Any,
			device: str = 'cpu',
			shuffle: bool = True,
			verbose: bool = False,
	):
		super(BaseTrainer, self).__init__()
		self.cfg = cfg
		self.shuffle = shuffle
		self.verbose = verbose
		self.device = torch.device(device)
		self.model = model.to(self.device).eval()
		self.stats = collections.defaultdict(dict)
		self.scaler = torch.cuda.amp.GradScaler(
			enabled=self.cfg.use_amp)
		self.model_ema = None
		self.ema_rate = None
		self.n_iters = None
		self.pbar = None

		self.writer = None
		self.wandb_run = None
		self._wandb = None
		self.logger = None
		self.dl_trn = None
		self.dl_vld = None
		self.dl_tst = None
		self.setup_data()

		self.optim = None
		self.optim_schedule = None
		self.setup_optim()

		if self.verbose:
			tot = sum([
				p.nelement() for p in
				self.model.parameters()
			])
			if tot // 1e6 > 0:
				tot = f"{tot / 1e6:0.1f} M"
			elif tot // 1e3 > 0:
				tot = f"{tot / 1e3:0.1f} K"
			print(f"\n# params: {tot}")

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
		if save:
			self.model.create_chkpt_dir(comment)
			self.cfg.save(self.model.chkpt_dir)
			writer = pjoin(
				self.model.cfg.runs_dir,
				os.path.basename(self.model.chkpt_dir),
			)
			self.writer = SummaryWriter(writer)
			self.logger = make_logger(
				name=type(self).__name__,
				path=self.model.chkpt_dir,
				level=logging.WARNING,
			)
			self.setup_wandb(comment)
		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule.T_max *= len(self.dl_trn)
		else:
			raise NotImplementedError

		self.pbar = tqdm(
			epochs,
			dynamic_ncols=True,
			leave=True,
			position=0,
		)
		try:
			for epoch in self.pbar:
				avg_loss = self.iteration(epoch, **kwargs)
				msg = ', '.join([
					f"epoch # {epoch + 1:d}",
					f"avg loss: {avg_loss:3f}",
				])
				self.pbar.set_description(msg)
				if not save:
					continue
				if (epoch + 1) % self.cfg.chkpt_freq == 0:
					self.save(
						checkpoint=epoch + 1,
						path=self.model.chkpt_dir,
					)
				if (epoch + 1) % self.cfg.eval_freq == 0:
					gstep = (epoch + 1) * len(self.dl_trn)
					_ = self.validate(gstep)
		finally:
			if self.writer is not None:
				self.writer.close()
			self.finish_wandb()
		return

	def iteration(self, epoch: int = 0, **kwargs):
		raise NotImplementedError

	def validate(self, epoch: int = None):
		raise NotImplementedError

	def setup_data(self):
		raise NotImplementedError

	def swap_model(self, new_model, full: bool = False):
		self.model = new_model.to(self.device).eval()
		if full:
			self.setup_data()
			self.setup_optim()
		return

	def select_model(self, ema: bool = False):
		if ema:
			assert self.model_ema is not None
			return self.model_ema.eval()
		return self.model.eval()

	def reset_model(self):
		pass

	def update_ema(self):
		if self.model_ema is None:
			return
		looper = zip(
			self.model.parameters(),
			self.model_ema.parameters(),
		)
		for p1, p2 in looper:
			p2.data.mul_(self.ema_rate)
			p2.data.add_(p1.data.mul(1-self.ema_rate))
		return

	def parameters(self, requires_grad: bool = True):
		if requires_grad:
			return filter(
				lambda p: p.requires_grad,
				self.model.parameters(),
			)
		else:
			return self.model.parameters()

	def save(self, path: str, checkpoint: int = None):
		if checkpoint is not None:
			global_step = checkpoint * len(self.dl_trn)
		else:
			global_step = None
		state_dict = {
			'metadata': {
				'checkpoint': checkpoint,
				'global_step': global_step,
				'stats': self.stats},
			'model': self.model.state_dict(),
			'model_ema': self.model_ema.state_dict()
			if self.model_ema is not None else None,
			'optim': self.optim.state_dict(),
			'scaler': self.scaler.state_dict(),
			'scheduler': self.optim_schedule.state_dict()
			if self.optim_schedule is not None else None,
		}
		fname = '+'.join([
			type(self.model).__name__,
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

	def setup_wandb(self, comment: str):
		if not getattr(self.cfg, 'use_wandb', False):
			return
		try:
			import wandb
		except ImportError:
			warnings.warn("wandb not installed; skipping W&B logging")
			return
		tags = list(getattr(self.cfg, 'wandb_tags', None) or [])
		tags += [
			type(self.model).__name__,
			type(self).__name__,
			getattr(self.model.cfg, 'sim', 'unknown-sim'),
			getattr(self.model.cfg, 'latent_type', 'unknown-latent'),
		]
		name = getattr(self.cfg, 'wandb_name', None) or comment
		self._wandb = wandb
		self.wandb_run = wandb.init(
			project=getattr(self.cfg, 'wandb_project', None),
			entity=getattr(self.cfg, 'wandb_entity', None),
			group=getattr(self.cfg, 'wandb_group', None),
			name=name,
			tags=sorted(set(tags)),
			mode=getattr(self.cfg, 'wandb_mode', 'online'),
			dir=self.model.chkpt_dir,
			config=self.wandb_config(),
		)
		self.wandb_run.summary.update(self.wandb_static_summary())
		self._save_wandb_configs()
		return

	def wandb_config(self):
		return {
			'model': self._config_dict(self.model.cfg),
			'train': self._config_dict(self.cfg),
			'derived': self.wandb_static_summary(),
		}

	def wandb_static_summary(self):
		n_params = sum(p.numel() for p in self.model.parameters())
		n_trainable = sum(
			p.numel() for p in self.model.parameters()
			if p.requires_grad
		)
		summary = {
			'model/name': self.model.cfg.name(),
			'model/params_total': n_params,
			'model/params_trainable': n_trainable,
			'train/n_iters': self.n_iters,
			'train/n_train_batches': len(self.dl_trn),
			'data/n_train': len(self.dl_trn.dataset),
			'data/n_val': len(self.dl_vld.dataset),
			'data/n_test': len(self.dl_tst.dataset),
		}
		if hasattr(self.model, 'total_latents'):
			summary['model/total_latents'] = self.model.total_latents()
		if hasattr(self.model.cfg, 'groups'):
			summary['model/latent_groups'] = list(self.model.cfg.groups)
			summary['model/n_latent_groups'] = sum(self.model.cfg.groups)
		return summary

	@staticmethod
	def _config_dict(cfg):
		out = {}
		for k, v in vars(cfg).items():
			if k.startswith('_'):
				continue
			if callable(v):
				continue
			if k in {'useful_yuwei'}:
				continue
			out[k] = BaseTrainer._jsonable(v)
		return out

	@staticmethod
	def _jsonable(v):
		if isinstance(v, (str, int, float, bool)) or v is None:
			return v
		if isinstance(v, np.generic):
			return v.item()
		if isinstance(v, torch.device):
			return str(v)
		if isinstance(v, pathlib.Path):
			return str(v)
		if isinstance(v, np.ndarray):
			return v.tolist()
		if isinstance(v, torch.Tensor):
			return to_np(v).tolist()
		if isinstance(v, tuple):
			return [BaseTrainer._jsonable(e) for e in v]
		if isinstance(v, list):
			return [BaseTrainer._jsonable(e) for e in v]
		if isinstance(v, dict):
			return {
				str(k): BaseTrainer._jsonable(val)
				for k, val in v.items()
			}
		return str(v)

	def _save_wandb_configs(self):
		if self.wandb_run is None:
			return
		for file_name in ['ConfigVAE.json', 'ConfigTrainVAE.json']:
			path = pjoin(self.model.chkpt_dir, file_name)
			if os.path.isfile(path):
				self.wandb_run.save(path, base_path=self.model.chkpt_dir)
		return

	def log_wandb_scalars(
			self,
			data: Dict[str, Any],
			step: int = None,
			commit: bool = True, ):
		if self.wandb_run is None:
			return
		payload = {}
		for k, v in data.items():
			v = self._to_scalar(v)
			if v is None:
				continue
			payload[k] = v
		if payload:
			self.wandb_run.log(payload, step=step, commit=commit)
		return

	def log_wandb_figures(
			self,
			figs: Dict[str, Any],
			step: int = None,
			commit: bool = True, ):
		if self.wandb_run is None or self._wandb is None:
			return
		payload = {
			k: self._wandb.Image(v)
			for k, v in figs.items()
		}
		if payload:
			self.wandb_run.log(payload, step=step, commit=commit)
		return

	@staticmethod
	def _to_scalar(v):
		if isinstance(v, np.generic):
			return v.item()
		if torch.is_tensor(v):
			if v.numel() != 1:
				return None
			return v.item()
		if isinstance(v, np.ndarray):
			if v.size != 1:
				return None
			return v.item()
		if isinstance(v, (int, float, bool)):
			return v
		return None

	def update_wandb_summary(self, data: Dict[str, Any]):
		if self.wandb_run is None:
			return
		for k, v in data.items():
			v = self._to_scalar(v)
			if v is not None:
				self.wandb_run.summary[k] = v
		return

	def finish_wandb(self):
		if self.wandb_run is None:
			return
		self._summarize_wandb_stats()
		self.wandb_run.finish()
		self.wandb_run = None
		self._wandb = None
		return

	def _summarize_wandb_stats(self):
		if self.wandb_run is None:
			return
		for k, vals in self.stats.items():
			if not isinstance(vals, dict) or not vals:
				continue
			steps = sorted(vals)
			a = [
				self._to_scalar(vals[s])
				for s in steps
			]
			a = np.array([v for v in a if v is not None], dtype=float)
			a = a[~np.isnan(a)]
			if not len(a):
				continue
			self.wandb_run.summary[f"last/{k}"] = a[-1]
			good = [
				'r2', 'mi', 'mig', 'r_aux', 'disentang', 'complete',
				'untangling',
			]
			if any(token in k for token in good):
				self.wandb_run.summary[f"best/{k}"] = np.nanmax(a)
			bad = ['loss', 'epe', 'kl', 'nelbo']
			if any(token in k for token in bad):
				self.wandb_run.summary[f"min/{k}"] = np.nanmin(a)
		return

	def setup_optim(self):
		# optimzer
		kws = dict(
			params=self.parameters(),
			lr=self.cfg.lr,
			**self.cfg.optimizer_kws,
		)
		if self.cfg.optimizer == 'adamax':
			self.optim = torch.optim.Adamax(**kws)
		elif self.cfg.optimizer == 'adamax_fast':
			from .adamax import Adamax
			self.optim = Adamax(**kws)
		elif self.cfg.optimizer == 'adam':
			self.optim = torch.optim.Adam(**kws)
		elif self.cfg.optimizer == 'adamw':
			self.optim = torch.optim.AdamW(**kws)
		elif self.cfg.optimizer == 'radam':
			self.optim = torch.optim.RAdam(**kws)
		else:
			raise NotImplementedError(self.cfg.optimizer)

		# scheduler
		if self.cfg.scheduler_type == 'cosine':
			self.optim_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'exponential':
			self.optim_schedule = torch.optim.lr_scheduler.ExponentialLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'step':
			self.optim_schedule = torch.optim.lr_scheduler.StepLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type == 'cyclic':
			self.optim = torch.optim.SGD(
				params=self.parameters(),
				lr=self.cfg.lr,
				momentum=0.9,
				weight_decay=self.cfg.optimizer_kws.get('weight_decay', 0),
			)
			self.optim_schedule = torch.optim.lr_scheduler.CyclicLR(
				self.optim, **self.cfg.scheduler_kws)
		elif self.cfg.scheduler_type is None:
			self.optim_schedule = None
		else:
			raise NotImplementedError(self.cfg.scheduler_type)
		return

	def to(self, x, dtype=torch.float32) -> Union[torch.Tensor, List[torch.Tensor]]:
		kws = dict(device=self.device, dtype=dtype)
		if isinstance(x, (tuple, list)):
			return [
				e.to(**kws) if torch.is_tensor(e)
				else torch.tensor(e, **kws)
				for e in x
			]
		else:
			if torch.is_tensor(x):
				return x.to(**kws)
			else:
				return torch.tensor(x, **kws)


def check_grads(
		grads: List[float],
		thres: float,
		gstep: int,
		fn: Callable = np.max, ):
	if fn(grads) > thres:
		msg = 'diverging grad encountered:\n'
		msg += f'fn: {fn.__name__}, grad: {fn(grads):0.1f} > {thres}'
		msg += f'\nglobal step = {gstep}, skipping . . . '
		print(msg)
		return True


def check_nans(loss, gstep: int, verbose: bool = True):
	if torch.isnan(loss).sum().item():
		msg = 'nan encountered in loss. '
		msg += 'optimizer will detect this & skip. '
		msg += f"global step = {gstep}"
		if verbose:
			print(msg)
		return True
	else:
		return False
