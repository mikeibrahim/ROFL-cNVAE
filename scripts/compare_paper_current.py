#!/usr/bin/env python
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from analysis.linear import mi_analysis, regress
from base.utils_model import load_model
from figures.dci import plot_bar_dci, plot_bar_untangle
from figures.fighelper import get_palette, prep_rofl
from figures.mi import plot_hm


OUT = ROOT / "outputs" / "paper_compare_current"
OUT.mkdir(parents=True, exist_ok=True)


MODEL_BASE = "fixate1_z-20x[3,6,12]_k-32_d-17_enc(2x2)-dec(2x1)-pre(1x3)-post(1x3)"
RUNS = [
	{
		"label": "cNVAE beta=0.15",
		"short": "cnvae_b015",
		"model": "cNVAE",
		"category": "fixate1",
		"beta": 0.15,
		"model_name": MODEL_BASE,
		"fit_name": "paper_best_lamb01_beta015_ep160-b600-lr(0.002)_beta(0.15:0x0.5)_lamb(0.01)_gr(250.0)_(2026_06_24,15:32)",
		"paper_fig4_r2": 0.898,
	},
	{
		"label": "cNVAE beta=0.5",
		"short": "cnvae_b05",
		"model": "cNVAE",
		"category": "fixate1",
		"beta": 0.5,
		"model_name": MODEL_BASE,
		"fit_name": "paper_best_lamb01_beta05_ep160-b600-lr(0.002)_beta(0.5:0x0.5)_lamb(0.01)_gr(250.0)_(2026_06_24,15:32)",
		"paper_fig4_r2": np.nan,
	},
	{
		"label": "P-cNVAE beta=0.15",
		"short": "pcnvae_b015",
		"model": "P-cNVAE",
		"category": "fixate1",
		"beta": 0.15,
		"model_name": f"{MODEL_BASE}_poisson",
		"fit_name": "paper_best_lamb01_beta015_poisson_cubic_ep160-b600-lr(0.002)_beta(0.15:0x0.5)_lamb(0.01)_gr(250.0)_(2026_06_24,15:32)",
		"paper_fig4_r2": np.nan,
	},
]


PAPER_PAGE_06 = OUT / "paper_300dpi-06.png"
PAPER_PAGE_07 = OUT / "paper_300dpi-07.png"
if PAPER_PAGE_06.exists() and PAPER_PAGE_07.exists():
	PAPER_CROPS = {
		"fig3_panel": (PAPER_PAGE_06, (1175, 823, 2130, 1308)),
		"fig3_full": (PAPER_PAGE_06, (1175, 795, 2210, 1770)),
		"fig4_panel": (PAPER_PAGE_07, (460, 790, 2180, 1188)),
		"fig4_full": (PAPER_PAGE_07, (435, 773, 2225, 1488)),
		"fig5_panel": (PAPER_PAGE_07, (1250, 1655, 2160, 2325)),
		"fig5_full": (PAPER_PAGE_07, (1250, 1625, 2210, 2538)),
	}
else:
	PAPER_CROPS = {
		"fig3_panel": (ROOT / "outputs" / "paper_pages" / "page-06.png", (470, 329, 852, 523)),
		"fig3_full": (ROOT / "outputs" / "paper_pages" / "page-06.png", (470, 318, 884, 708)),
		"fig4_panel": (ROOT / "outputs" / "paper_pages" / "page-07.png", (184, 316, 872, 475)),
		"fig4_full": (ROOT / "outputs" / "paper_pages" / "page-07.png", (174, 309, 890, 595)),
		"fig5_panel": (ROOT / "outputs" / "paper_pages" / "page-07.png", (500, 662, 864, 930)),
		"fig5_full": (ROOT / "outputs" / "paper_pages" / "page-07.png", (500, 650, 884, 1015)),
	}


def patch_torch_load():
	orig_load = torch.load

	def load(*args, **kwargs):
		kwargs.setdefault("weights_only", False)
		return orig_load(*args, **kwargs)

	torch.load = load


def safe_name(s):
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def save_npz(path, **kwargs):
	np.savez_compressed(path, **kwargs)


def load_or_compute_run(run, g):
	cache = OUT / f"{run['short']}_selected_eval.npz"
	mi_cache = OUT / f"{run['short']}_selected_mi.npz"
	if cache.exists():
		data = dict(np.load(cache, allow_pickle=True))
	else:
		print(f"load/regress {run['label']}", flush=True)
		trainer, meta = load_model(
			model_name=run["model_name"],
			fit_name=run["fit_name"],
			checkpoint=-1,
			device="cuda:0" if torch.cuda.is_available() else "cpu",
			verbose=False,
		)
		regr = trainer.regress()
		sel = regress(
			z=regr["z_vld"],
			z_tst=regr["z_tst"],
			g=g["vld"],
			g_tst=g["tst"],
		)
		data = {
			"z_vld": regr["z_vld"],
			"z_tst": regr["z_tst"],
			"selected_r2": sel["r2"],
			"selected_d": np.array(sel["d"]),
			"selected_c": np.array(sel["c"]),
			"all_r2": regr["regr/r2"],
			"aux_r2": regr["regr/aux/r2"],
			"all_d": np.array(regr["regr/d"]),
			"all_c": np.array(regr["regr/c"]),
			"checkpoint": np.array(meta.get("file", "")),
			"fit_path": np.array(meta.get("path", "")),
		}
		save_npz(cache, **data)
		del trainer
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
	if mi_cache.exists():
		mi = dict(np.load(mi_cache, allow_pickle=True))
	else:
		print(f"mi {run['label']}", flush=True)
		mi = mi_analysis(
			z=data["z_vld"],
			g=g["vld"],
			parallel=True,
			n_jobs=12,
		)
		save_npz(mi_cache, **mi)
	return data, mi


def crop_paper_panels():
	paths = {}
	for name, (src, box) in PAPER_CROPS.items():
		dst = OUT / f"paper_{name}.png"
		img = Image.open(src).convert("RGB")
		img.crop(box).save(dst)
		paths[name] = dst
	return paths


def pad_to_width(img, width):
	if img.width == width:
		return img
	canvas = Image.new("RGB", (width, img.height), "white")
	canvas.paste(img, ((width - img.width) // 2, 0))
	return canvas


def resize_height(img, height):
	if img.height == height:
		return img
	width = int(round(img.width * height / img.height))
	return img.resize((width, height), Image.Resampling.LANCZOS)


def side_by_side(left, right, dst, title_left="paper", title_right="ours"):
	left = Image.open(left).convert("RGB")
	right = Image.open(right).convert("RGB")
	height = max(left.height, right.height)
	left = resize_height(left, height)
	right = resize_height(right, height)
	top = 44
	gap = 22
	canvas = Image.new("RGB", (left.width + right.width + gap, height + top), "white")
	draw = ImageDraw.Draw(canvas)
	draw.text((8, 10), title_left, fill="black")
	draw.text((left.width + gap + 8, 10), title_right, fill="black")
	canvas.paste(left, (0, top))
	canvas.paste(right, (left.width + gap, top))
	canvas.save(dst)


def write_metrics(rows):
	df = pd.DataFrame(rows)
	df.to_csv(OUT / "metrics_selected.csv", index=False)
	with open(OUT / "metrics_selected.json", "w") as f:
		json.dump(rows, f, indent=2)
	return df


def make_fig3(mi_by_run):
	data = {
		"cNVAE beta=0.15": mi_by_run["cnvae_b015"]["mi"],
		"P-cNVAE beta=0.15": mi_by_run["pcnvae_b015"]["mi"],
	}
	fig, axes = plot_hm(data, display=False)
	axes = np.ravel(axes)
	for ax, label, color in zip(
			axes,
			["cNVAE (ours)", "P-cNVAE (ours)"],
			["#5f79bd", "#d58442"]):
		ax.annotate(
			label,
			xy=(1.01, 0.5),
			xycoords="axes fraction",
			rotation=-90,
			va="center",
			ha="left",
			fontsize=12,
			fontweight="bold",
			color=color,
		)
	fig.savefig(OUT / "fig3_ours_mi_repo_plot_hm.png", dpi=300, bbox_inches="tight")
	fig.savefig(OUT / "fig3_ours_mi_repo_plot_hm.pdf", bbox_inches="tight")
	plt.close(fig)


def make_fig4(results, select_lbl):
	pal = get_palette()[0]
	row = results["cnvae_b015"]
	df = pd.DataFrame({
		"f": select_lbl,
		"r2": row["data"]["selected_r2"],
		"model": ["cNVAE"] * len(select_lbl),
	})
	fig, ax = plot_bar_untangle(df, pal=pal, display=False)
	ax.set_title("ours: cNVAE beta=0.15, selected-label mean R2 = "
				 f"{np.nanmean(row['data']['selected_r2']):0.3f}", fontsize=11)
	fig.savefig(OUT / "fig4_ours_cnvae_repo_plot_bar_untangle.png", dpi=300, bbox_inches="tight")
	fig.savefig(OUT / "fig4_ours_cnvae_repo_plot_bar_untangle.pdf", bbox_inches="tight")
	plt.close(fig)


def make_fig5(rows):
	pal = get_palette()[0]
	df = pd.DataFrame([
		{
			"category": row["category"],
			"beta": row["beta"],
			"model": row["model"],
			"i": row["selected_r2_mean"],
			"d": row["selected_d"],
			"c": row["selected_c"],
		}
		for row in rows
		if row["model"] == "cNVAE"
	])
	fig, axes = plot_bar_dci(df, cat="fixate1", pal=pal, display=False)
	fig.savefig(OUT / "fig5_ours_cnvae_repo_plot_bar_dci.png", dpi=300, bbox_inches="tight")
	fig.savefig(OUT / "fig5_ours_cnvae_repo_plot_bar_dci.pdf", bbox_inches="tight")
	plt.close(fig)


def main():
	patch_torch_load()
	paper_paths = crop_paper_panels()
	g, _, select_lbl = prep_rofl("fixate1")
	results = {}
	mi_by_run = {}
	rows = []
	for run in RUNS:
		data, mi = load_or_compute_run(run, g)
		results[run["short"]] = {"run": run, "data": data, "mi": mi}
		mi_by_run[run["short"]] = mi
		selected_r2_mean = float(np.nanmean(data["selected_r2"]))
		paper = run["paper_fig4_r2"]
		rows.append({
			"label": run["label"],
			"model": run["model"],
			"category": run["category"],
			"beta": run["beta"],
			"selected_r2_mean": selected_r2_mean,
			"paper_fig4_r2": None if np.isnan(paper) else paper,
			"delta_vs_paper_fig4": None if np.isnan(paper) else selected_r2_mean - paper,
			"selected_d": float(data["selected_d"]),
			"selected_c": float(data["selected_c"]),
			"all_g_r2_mean_train_loop_basis": float(np.nanmean(data["all_r2"])),
			"aux_r2_mean_train_loop_basis": float(np.nanmean(data["aux_r2"])),
			"mi_max_mean_selected": float(np.nanmax(mi["mi"], axis=1).mean()),
			"mi_norm_max_mean_selected": float(np.nanmax(mi["mi_norm"], axis=1).mean()),
			"mig_mean_selected": float(np.nanmean(mi["mig"])),
			"checkpoint": str(data["checkpoint"]),
			"fit_path": str(data["fit_path"]),
		})
	df = write_metrics(rows)
	make_fig3(mi_by_run)
	make_fig4(results, select_lbl)
	make_fig5(rows)
	side_by_side(
		paper_paths["fig3_panel"],
		OUT / "fig3_ours_mi_repo_plot_hm.png",
		OUT / "compare_fig3_paper_vs_ours.png",
		"paper Fig.3",
		"ours: repo plot_hm",
	)
	side_by_side(
		paper_paths["fig4_panel"],
		OUT / "fig4_ours_cnvae_repo_plot_bar_untangle.png",
		OUT / "compare_fig4_paper_vs_ours.png",
		"paper Fig.4",
		"ours: repo plot_bar_untangle",
	)
	side_by_side(
		paper_paths["fig5_panel"],
		OUT / "fig5_ours_cnvae_repo_plot_bar_dci.png",
		OUT / "compare_fig5_paper_vs_ours.png",
		"paper Fig.5",
		"ours: repo plot_bar_dci",
	)
	print(df.to_string(index=False), flush=True)
	print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
	main()
