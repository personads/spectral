#!/usr/bin/python3

import argparse, os, sys

from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.datasets import LabelledDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Evaluate Sentence-level Classification')
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
	targets, predictions = [], []
	for idx in range(tgt_data.get_input_count()):
		tgt_text, tgt_label = tgt_data[idx]
		prd_text, prd_labels = prd_data[idx]

		if type(prd_labels) is list:
			tgt_labels = [tgt_label for _ in range(len(prd_labels))]
		else:
			tgt_labels, prd_labels = [tgt_label], [prd_labels]

		num_correct += sum([1 for tl, pl in zip(tgt_labels, prd_labels) if tl == pl])
		num_total += len(tgt_labels)

		targets += tgt_labels
		predictions += prd_labels

	accuracy = num_correct/num_total
	print(f"Accuracy: {accuracy * 100:.2f}% ({num_correct}/{num_total})")

	f1_macro = f1_score(targets, predictions, average='macro')
	print(f"F1 (macro): {f1_macro * 100:.2f}%")

	f1_micro = f1_score(targets, predictions, average='micro')
	print(f"F1 (micro): {f1_micro * 100:.2f}%")


if __name__ == '__main__':
	main()
