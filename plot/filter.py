#!/usr/bin/python3

import argparse, os, sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.encoders import *
from plot.decomposition import plot as plot_decomposition


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Embed Data')
	arg_parser.add_argument('model_path', help='model name in the transformers library or path to model')
	arg_parser.add_argument('output_prefix', help='prefix path for output plots')
	arg_parser.add_argument('-t', '--text', help='input text to process for decomposition plot')
	return arg_parser.parse_args()


def encode(encoder, text):
	emb_tokens, _ = encoder([text])
	embeddings = emb_tokens[0].tolist()
	tokens = encoder._tok.convert_ids_to_tokens(encoder.tokenize([text])['input_ids'][0], skip_special_tokens=True)

	return tokens, embeddings


def plot_filter(frq_filter, path=None):
	# plot bars (A4: 210x240mm, width with margins = 6.3in)
	colormap = mpl.cm.get_cmap('rainbow_r')
	mapnorm = mpl.colors.Normalize(vmin=0, vmax=frq_filter.shape[0])
	fig, ax = plt.subplots(figsize=(6.3 * 1.5, 1.75))

	# set up x-axis
	x_range = np.arange(frq_filter.shape[0])
	ax.set_xticks([0, (frq_filter.shape[0] - 1)//2, frq_filter.shape[0] - 1])
	ax.set_xlabel('Frequencies', fontsize='large', alpha=.6)
	ax.set_xlim(0, frq_filter.shape[0] - 1)

	# set up y-axis
	ax.set_ylim(0, 1)
	ax.set_ylabel('Weight', fontsize='large', alpha=.6)

	caption = ''

	# plot scores and stddev
	bars = ax.bar(x_range, frq_filter, alpha=.7)
	# set colors
	for bidx, bar in enumerate(bars):
		bar.set_color(colormap(mapnorm(bidx)))

	# add grid
	ax.grid(axis='y', linestyle=':', linewidth=1.5)
	ax.set_axisbelow(True)

	for lidx in x_range:
		caption += f'{frq_filter[lidx]:.1f}, '
	caption = caption[:-2] + '. '

	print(f"Screenreader Caption: {caption}")

	fig.tight_layout()
	if path is None:
		plt.show()
	else:
		plt.savefig(path)
		print(f"Saved filter plot to '{path}'.")


def main():
	args = parse_arguments()

	# load model
	encoder = PrismEncoder.load(args.model_path)
	# check CUDA availability
	if torch.cuda.is_available(): encoder.to(torch.device('cuda'))
	# set encoder to inference mode
	encoder.eval()

	# plot frequency filter
	scaled_filter = torch.sigmoid(encoder._frq_filter.detach()).cpu().numpy()
	plot_filter(scaled_filter, path=args.output_prefix + '-filter.pdf')

	# plot example text decomposition
	if not args.text:
		exit()

	# get embeddings from autofilter encoder
	tokens, emb_filtered = encode(encoder, args.text)

	# get embeddings from unfiltered model
	encoder = PrismEncoder(lm_name=torch.load(args.model_path)['language_model_name'], frq_filter=None, cache={})
	# check CUDA availability
	if torch.cuda.is_available(): encoder.to(torch.device('cuda'))
	# set encoder to inference mode
	encoder.eval()
	_, emb_unfiltered = encode(encoder, args.text)

	# plot frequencies of individual neuron
	embeddings = np.array([emb_unfiltered, emb_filtered])
	plot_decomposition(tokens, embeddings, [0, 127, 255, 511, 767], path=args.output_prefix + '-dec.pdf')


if __name__ == '__main__':
	main()
