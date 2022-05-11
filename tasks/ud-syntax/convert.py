#!/usr/bin/python3

import argparse, csv, os

from ud import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Dataset Conversion (Syntax)')
	arg_parser.add_argument('input_path', help='path to Universal Dependencies treebank directory')
	arg_parser.add_argument('task', choices=['relations', 'pos'], help='task identifier')
	arg_parser.add_argument('out_path', help='path to CSV output directory')
	return arg_parser.parse_args()


def load_treebanks(path):
	treebanks = {}

	# iterate over files in TB directory
	for tbf in sorted(os.listdir(path)):
		# skip non-conllu files
		if os.path.splitext(tbf)[1] != '.conllu': continue

		# extract treebank name (e.g. 'en-ewt-dev')
		tb_name = os.path.splitext(tbf)[0].replace('-ud-', '-').replace('_', '-')

		# load treebank
		tbf_path = os.path.join(path, tbf)
		treebank = UniversalDependencies(treebanks=[UniversalDependenciesTreebank.from_conllu(tbf_path, name=tbf)])
		treebanks[tb_name] = treebank
		print(f"Loaded {treebank}.")

	return treebanks


def main():
	args = parse_arguments()

	# load treebank splits and relation classes
	treebanks = load_treebanks(args.input_path)
	print(f"Loaded {len(treebanks)} treebank splits from '{args.input_path}'.")

	# write splits to files
	for tb_name, treebank in treebanks.items():
		split_path = os.path.join(args.out_path, f'{tb_name}-{args.task}.csv')
		label_values = set()
		# write to CSV file
		with open(split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text', 'label'])
			# iterate over sentences
			for sidx in range(len(treebank)):
				# retrieve tokenized sentence
				words = treebank[sidx].to_words()
				words = [w.replace(' ', '') for w in words]  # remove spaces within words (e.g. FR-GSD)
				# retrieve relevant labels
				if args.task == 'relations':
					_, labels = treebank[sidx].get_dependencies(include_subtypes=False)
				elif args.task == 'pos':
					labels = treebank[sidx].get_pos()
				# store labels
				label_values |= set(labels)
				# prepare row
				text = ' '.join(words)
				labels = ' '.join(labels)
				# write row to file
				csv_writer.writerow([text, labels])
		print(f"Saved {tb_name} with {sidx + 1} sentences and {len(label_values)} label types to '{split_path}'.")


if __name__ == '__main__':
	main()
