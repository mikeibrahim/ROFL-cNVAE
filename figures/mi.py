from .fighelper import *


def plot_hm(
		data: Dict[str, np.ndarray],
		scale_factor: int = 25,
		cmap: str = 'bone_r',
		groups: List[int] = None,
		latent_per_group: int = None,
		spatial_scales: List[int] = None,
		labels: List[str] = None,
		model_labels: Dict[str, str] = None,
		model_colors: Dict[str, str] = None,
		show_scale_brackets: bool = True,
		display: bool = True, ):
	assert scale_factor % 2 == 1, "must be an odd number"
	assert len(data), "data must contain at least one MI matrix"
	groups = [3, 6, 12] if groups is None else list(groups)
	spatial_scales = [2, 4, 8] if spatial_scales is None \
		else list(spatial_scales)
	labels = list(LBL2TEX.values()) if labels is None else list(labels)
	assert len(groups) == len(spatial_scales)
	n_latents = {fit: v.shape[1] for fit, v in data.items()}
	if latent_per_group is None:
		lpg = {fit: n // sum(groups) for fit, n in n_latents.items()}
	else:
		lpg = {fit: latent_per_group for fit in data}
	for fit, v in data.items():
		assert v.ndim == 2
		assert v.shape[0] == len(labels)
		assert v.shape[1] == sum(groups) * lpg[fit]
	nrows = len(data)
	sharex = len(set(n_latents.values())) == 1
	fig, axes = create_figure(
		nrows=nrows,
		ncols=1,
		figsize=(10, 1.85 * nrows + 0.55),
		sharex='all' if sharex else False,
		sharey='all',
		layout='constrained',
	)
	axes = np.atleast_1d(axes)
	for i, (fit, v) in enumerate(data.items()):
		x2p = np.repeat(np.repeat(
			v, scale_factor, axis=0),
			scale_factor, axis=1)
		axes[i].imshow(
			X=x2p,
			aspect=9 * n_latents[fit] / 420,
			vmin=0,
			vmax=0.5,
			cmap=cmap,
		)
		group_width = lpg[fit] * scale_factor
		for j in range(1, sum(groups)):
			axes[i].axvline(
				x=j * group_width - 0.5,
				color='dimgrey',
				alpha=0.7,
				ls='--',
				lw=0.7,
			)
		ticks = np.arange(sum(groups) + 1) * group_width
		axes[i].set(
			yticks=np.arange(len(labels)) * scale_factor + scale_factor // 2,
			yticklabels=labels,
			xticks=ticks,
			xticklabels=np.arange(sum(groups) + 1) * lpg[fit],
		)
		axes[i].tick_params(axis='x', labelbottom=True)
		if model_labels is not None:
			label = model_labels.get(fit, fit)
			color = 'black' if model_colors is None \
				else model_colors.get(fit, 'black')
			axes[i].annotate(
				label,
				xy=(1.01, 0.5),
				xycoords='axes fraction',
				rotation=-90,
				va='center',
				ha='left',
				fontsize=12,
				fontweight='bold',
				color=color,
			)
		if show_scale_brackets:
			_add_scale_brackets(
				axes[i],
				groups=groups,
				spatial_scales=spatial_scales,
				group_width=group_width,
			)
	if display:
		plt.show()
	else:
		plt.close()
	return fig, axes


def _add_scale_brackets(
		ax,
		groups: List[int],
		spatial_scales: List[int],
		group_width: int, ):
	trans = matplotlib.transforms.blended_transform_factory(
		ax.transData, ax.transAxes)
	start = 0
	for n_groups, scale in zip(groups, spatial_scales):
		stop = start + n_groups * group_width
		pad = 0.25 * group_width
		x0, x1 = start + pad, stop - pad
		ax.plot(
			[x0, x0, x1, x1],
			[1.05, 1.13, 1.13, 1.05],
			color='black',
			lw=0.9,
			transform=trans,
			clip_on=False,
		)
		ax.text(
			(x0 + x1) / 2,
			1.15,
			rf'${scale} \times {scale}$',
			ha='center',
			va='bottom',
			fontsize=13,
			transform=trans,
		)
		start = stop
	return
