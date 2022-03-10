#!/usr/bin/python3

import argparse, csv, os, re, sys

from collections import defaultdict

import numpy as np


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='20 Newsgroups - Dataset Conversion')
	arg_parser.add_argument('input_path', help='path to 20 Newsgroups directory (containing splits)')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpus')
	arg_parser.add_argument('-vp', '--validation_proportion', type=float, default=.2, help='proportion of training data to use as validation data (default: .2)')
	arg_parser.add_argument('-rs', '--random_seed', type=int, help='seed for probabilistic components (default: None)')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# set random seed
	np.random.seed(args.random_seed)

	# initialize output lines
	split_data = defaultdict(list)

	# iterate over split directories
	for split_path in os.listdir(args.input_path):
		split_path = os.path.join(args.input_path, split_path)
		if (not os.path.isdir(split_path)) or (len(split_path.split('-')) != 3):
			continue
		split_name = os.path.basename(split_path.split('-')[-1])

		# iterate over newsgroups
		for class_name in os.listdir(split_path):
			class_path = os.path.join(split_path, class_name)
			if not os.path.isdir(class_path):
				continue
			# iterate over texts
			for text_path in os.listdir(class_path):
				if re.match(r'\d+', text_path) is None:
					continue
				text_path = os.path.join(class_path, text_path)
				# load full text
				with open(text_path, 'r', encoding='iso-8859-1') as fop:
					text = fop.read()
					label = class_name

					split_data[split_name].append([text, label])

		print(f"Loaded '{split_name}' data with {len(split_data[split_name])} items.")

	# split off validation data from training
	split_idx = round(len(split_data['train']) * args.validation_proportion)
	# shuffle the data in-place
	np.random.shuffle(split_data['train'])
	# create validation split
	split_data['dev'] = split_data['train'][:split_idx]
	# discard from training split
	split_data['train'] = split_data['train'][split_idx:]
	# re-sort data by labels
	split_data['train'].sort(key=lambda el: el[1])
	split_data['dev'].sort(key=lambda el: el[1])
	print(f"Created random splits: {', '.join([f'{s}: {len(lines)}' for s, lines in split_data.items()])} using random seed {args.random_seed}.")

	# write splits to files
	for split, lines in split_data.items():
		split_path = args.output_path + f'-{split}.csv'
		with open(split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text', 'label'])
			csv_writer.writerows(lines)
		print(f"Saved {split}-split with {len(lines)} items to '{split_path}'.")


if __name__ == '__main__':
	main()
