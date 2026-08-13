#!/usr/bin/env python
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.linear import mi_analysis
from base.utils_model import load_model
from figures.fighelper import LBL2TEX, prep_rofl
from figures.mi import plot_hm


OUT = ROOT / 'outputs' / 'mi_latents_per_group_paper_style'
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
	20: {
		'model_name': 'fixate1_z-20x[3,6,12]_k-32_d-17_enc(2x2)-dec(2x1)-pre(1x3)-post(1x3)',
		'fit_name': 'paper_best_lamb01_beta015_ep160-b600-lr(0.002)_beta(0.15:0x0.5)_lamb(0.01)_gr(250.0)_(2026_06_24,15:32)',
	},
	14: {
		'model_name': 'fixate1_z-14x[3,6,12]_k-32_d-17_enc(2x2)-dec(2x1)-pre(1x3)-post(1x3)',
		'fit_name': 'gbs_v1_lpg14_z294_ch32_ep160-b600-lr(0.002)_beta(0.15:0x0.5)_lamb(0.01)_gr(250.0)_(2026_07_29,07:08)',
	},
	8: {
		'model_name': 'fixate1_z-8x[3,6,12]_k-32_d-17_enc(2x2)-dec(2x1)-pre(1x3)-post(1x3)',
		'fit_name': 'gbs_v2_lpg8_z168_ch32_ep160-b600-lr(0.002)_beta(0.15:0x0.5)_lamb(0.01)_gr(250.0)_(2026_08_12,20:54)',
	},
}


def patch_torch_load():
	original = torch.load

	def load(*args, **kwargs):
		kwargs.setdefault('weights_only', False)
		return original(*args, **kwargs)

	torch.load = load


def load_or_compute_mi(latents_per_group, run, g, device):
	cache = OUT / f'mi_{latents_per_group}_per_group.npz'
	if cache.exists():
		return dict(np.load(cache))
	if latents_per_group == 20:
		paper_cache = ROOT / 'outputs' / 'paper_compare_current' / \
			'cnvae_b015_selected_mi.npz'
		if paper_cache.exists():
			mi = dict(np.load(paper_cache))
			np.savez_compressed(cache, **mi)
			return mi
	trainer, _ = load_model(
		model_name=run['model_name'],
		fit_name=run['fit_name'],
		checkpoint=-1,
		device=device,
		verbose=False,
	)
	regr = trainer.regress()
	mi = mi_analysis(
		z=regr['z_vld'],
		g=g['vld'],
		parallel=True,
		n_jobs=12,
	)
	np.savez_compressed(cache, **mi)
	del trainer
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	return mi


def save_plot(latents_per_group, mi):
	key = f'lpg{latents_per_group}'
	fig, _ = plot_hm(
		{key: mi['mi']},
		groups=[3, 6, 12],
		latent_per_group=latents_per_group,
		spatial_scales=[2, 4, 8],
		labels=list(LBL2TEX.values()),
		model_labels={key: f'cNVAE ({latents_per_group}/group)'},
		model_colors={key: '#5f79bd'},
		display=False,
	)
	stem = OUT / f'mi_{latents_per_group}_per_group_paper_style'
	fig.savefig(stem.with_suffix('.png'), dpi=300, bbox_inches='tight')
	fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
	plt.close(fig)


def main():
	patch_torch_load()
	g, _, _ = prep_rofl('fixate1')
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	for latents_per_group, run in RUNS.items():
		print(f'processing {latents_per_group}/group', flush=True)
		mi = load_or_compute_mi(latents_per_group, run, g, device)
		save_plot(latents_per_group, mi)
	print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
	main()
