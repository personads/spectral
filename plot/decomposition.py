#!/usr/bin/python3

import argparse, os, sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.encoders import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Embed Data')
	arg_parser.add_argument('in_text', help='input text')
	arg_parser.add_argument('lm_name', help='model name in the transformers library or path to model')
	arg_parser.add_argument('num_bands', type=int, help='number of frequency bands to split embeddings into')
	return arg_parser.parse_args()


def encode(args):
	# set up filter with (almost) equal allocation to all bands
	lm_config = transformers.AutoConfig.from_pretrained(args.lm_name)
	filters = PrismEncoder.gen_equal_allocation_filters(lm_config.max_position_embeddings, args.num_bands)

	# load encoder
	encoder = PrismEncoder(lm_name=args.lm_name, frq_filter=None, cache={})
	# check CUDA availability
	if torch.cuda.is_available(): encoder.to(torch.device('cuda'))
	# set encoder to inference mode
	encoder.eval()
	print(f"Loaded '{args.lm_name}':")
	print(encoder)

	# initialize output
	embeddings = []
	tokens = []

	# perform unfiltered pass
	emb_tokens, _ = encoder([args.in_text])
	embeddings.append(emb_tokens[0].tolist())
	tokens = encoder._tok.convert_ids_to_tokens(encoder.tokenize([args.in_text])['input_ids'][0], skip_special_tokens=True)
	print(f"[0/{args.num_bands}] Encoded without filter.")

	# iterate over each frequency band
	for bidx in range(args.num_bands):
		# set filter
		encoder._frq_filter = filters[bidx]

		# encode sentence
		emb_tokens, _ = encoder([args.in_text])
		embeddings.append(emb_tokens[0].tolist())
		print(f"[{bidx+1}/{args.num_bands}] Encoded with filter 'eqalloc({lm_config.max_position_embeddings}, {args.num_bands}, {bidx})'.")

	return tokens, np.array(embeddings)


def plot(tokens, emb_bands, emb_dims, path=None):
	# set up figure
	fig, axs = plt.subplots(len(emb_dims), 1, figsize=(6.3, 1.75 * len(emb_dims)))
	colormap = mpl.cm.get_cmap('rainbow_r')
	mapnorm = mpl.colors.Normalize(vmin=1, vmax=emb_bands.shape[0])

	# iterate over relevant embedding dimensions
	for eidx, emb_dim in enumerate(emb_dims):
		# reduce to single neuron activations
		embeddings = emb_bands[:, :, emb_dim]

		ax = axs[eidx]
		# set up x-axis
		x_range = np.arange(embeddings.shape[1])
		ax.set_xticks(x_range)
		ax.set_xlim(0, embeddings.shape[1] - 1)
		if eidx == len(emb_dims) - 1:
			ax.set_xticklabels(tokens, rotation=90)
		else:
			ax.tick_params(bottom=False, labelbottom=False)
		# ax.set_xlabel('Token', fontsize='large', alpha=.6)

		# set up y-axis
		ax.set_ylim(np.min(emb_bands[:, :, emb_dims]), np.max(emb_bands[:, :, emb_dims]))
		ax.set_ylabel(f'Activation {emb_dim}', alpha=.6)

		# plot unfiltered
		ax.plot(x_range, embeddings[0], color='darkgrey', alpha=.5, label='no filter')

		# plot each band
		for bidx in range(1, embeddings.shape[0]):
			ax.plot(x_range, embeddings[bidx], color=colormap(mapnorm(bidx)), alpha=.7, label=f'band {bidx}')

		ax.grid(linestyle=':', linewidth=1)
		ax.set_axisbelow(True)

	# ax.legend()
	fig.tight_layout()
	if path is None:
		plt.show()
	else:
		plt.savefig(path)
		print(f"Saved decomposition plot to '{path}'.")


def main():
	args = parse_arguments()

	# get embeddings from each frequency band (1 + num_bands, seq_len, emb_dim)
	tokens, embeddings = encode(args)

	# plot frequencies of individual neuron
	plot(tokens, embeddings, [0, 127, 255, 511, 767])


if __name__ == '__main__':
	main()
