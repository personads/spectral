#!/usr/bin/python3

import argparse, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.datasets import LabelledDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='20 Newsgroups - Evaluate Accuracy')
	arg_parser.add_argument('tgt_path', help='path to target CSV')
	arg_parser.add_argument('prd_path', help='path to predicted CSV')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	tgt_data = LabelledDataset.from_path(args.tgt_path)
	print(f"Loaded target data {tgt_data} from '{args.tgt_path}'.")
	prd_data = LabelledDataset.from_path(args.prd_path)
	print(f"Loaded predicted data {prd_data} from '{args.prd_path}'.")

	num_correct, num_total = 0, 0
	for idx in range(tgt_data.get_input_count()):
		tgt_text, tgt_label = tgt_data[idx]
		prd_text, prd_labels = prd_data[idx]

		tgt_labels = [tgt_label for _ in range(len(prd_labels))]

		num_correct += sum([1 for tl, pl in zip(tgt_labels, prd_labels) if tl == pl])
		num_total += len(tgt_labels)

	accuracy = num_correct/num_total
	print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
	main()
