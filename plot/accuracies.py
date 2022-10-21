#!/usr/bin/python3

import argparse, json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Plot Accuracies per Frequency Band')
	arg_parser.add_argument('accuracies', nargs='+', help='lists of accuracies per run (e.g. across initializations)')
	arg_parser.add_argument('-xl', '--x_labels', nargs='+', help="names of frequency bands")
	arg_parser.add_argument('-hx', '--hide_xlabel', action='store_true', default=False, help="flag to hide x-axis label")
	arg_parser.add_argument('-hy', '--hide_ylabel', action='store_true', default=False, help="flag to hide y-axis label")
	arg_parser.add_argument('-op', '--output_path', help="path to output file")
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# load experiment scores
	experiments = []
	for eidx, experiment in enumerate(args.accuracies):
		experiments.append(json.loads(experiment))
	experiments = np.array(experiments)
	experiments = experiments * 100 if np.max(experiments) <= 1 else experiments
	print(f"Loaded {experiments.shape[1]} experiments across {experiments.shape[0]} initializations.")

	# plot bars (A4: 210x240mm, width with margins = 6.3in)
	colormap = mpl.cm.get_cmap('rainbow_r')
	# colormap = mpl.cm.get_cmap('coolwarm_r')
	mapnorm = mpl.colors.Normalize(vmin=1, vmax=experiments.shape[1]-1)
	fig, ax = plt.subplots(figsize=(6.3 * .5, 6.3 * .4))

	ax.tick_params(axis='both', labelsize='medium')

	# set up x-axis
	x_range = np.arange(experiments.shape[1])
	ax.set_xticks(np.arange(experiments.shape[1]), args.x_labels)
	if not args.hide_xlabel:
		ax.set_xlabel('Frequency Bands', fontsize='x-large', alpha=.6)

	# set up y-axis
	ax.set_ylim(0, 100)
	if not args.hide_ylabel:
		ax.set_ylabel('Accuracy', fontsize='x-large', alpha=.6)

	caption = ''
	# calculate mean and stddev
	mean_scores = np.mean(experiments, axis=0)
	stddev_scores = np.std(experiments, axis=0)

	# plot scores and stddev
	bars = ax.bar(x_range, mean_scores, alpha=.7)
	# set colors
	for bidx, bar in enumerate(bars):
		if bidx == 0:
			bar.set_color('darkgray')
		elif bidx == len(bars) - 1:
			bar.set_color('slateblue')
		else:
			bar.set_color(colormap(mapnorm(bidx)))

	# add grid
	ax.grid(axis='y', linestyle=':', linewidth=1.5)
	ax.set_axisbelow(True)

	for lidx in x_range:
		caption += f'{mean_scores[lidx]:.1f}, '
	caption = caption[:-2] + '. '

	print(f"Screenreader Caption: {caption}")

	fig.tight_layout()
	if args.output_path is not None:
		plt.savefig(args.output_path, bbox_inches='tight', pad_inches=.05)
	plt.show()


if __name__ == '__main__':
	main()
