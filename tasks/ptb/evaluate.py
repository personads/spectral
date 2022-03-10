#!/usr/bin/python3

import argparse, os, sys

import transformers

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.datasets import LabelledDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Penn Treebank - Evaluate Accuracy')
	arg_parser.add_argument('tgt_path', help='path to target CSV')
	arg_parser.add_argument('prd_path', help='path to predicted CSV')
	arg_parser.add_argument('-t', '--tokenizer', help='name of HuggingFace tokenizer')
	return arg_parser.parse_args()


def repeat_labels(texts, labels, tokenizer):
	rep_labels = []
	lbl_cursor = -1
	for idx in range(len(texts)):
		tok_text = tokenizer(
			texts[idx].split(' '),
			padding=True, truncation=True,
			return_special_tokens_mask=True, return_offsets_mapping=True, is_split_into_words=True
		)
		for tidx in range(sum(tok_text['attention_mask'])):
			if tok_text['special_tokens_mask'][tidx] == 1: continue

			# check for start of new token
			if tok_text['offset_mapping'][tidx][0] == 0:
				lbl_cursor += 1
			rep_labels.append(labels[lbl_cursor])
	return rep_labels


def main():
	args = parse_arguments()

	tgt_data = LabelledDataset.from_path(args.tgt_path)
	print(f"Loaded target data {tgt_data} from '{args.tgt_path}'.")
	prd_data = LabelledDataset.from_path(args.prd_path)
	print(f"Loaded predicted data {prd_data} from '{args.prd_path}'.")

	if args.tokenizer:
		tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, add_prefix_space=True)
		print(f"Loaded '{args.tokenizer}' ({tokenizer.__class__.__name__}) for label repetition.")

	num_correct, num_total = 0, 0
	for tgt_batch, prd_batch in zip(tgt_data.get_batches(100), prd_data.get_batches(100)):
		tgt_text, tgt_labels, _ = tgt_batch
		prd_text, prd_labels, _ = prd_batch

		if args.tokenizer:
			tgt_labels = repeat_labels(tgt_text, tgt_labels, tokenizer)
		assert len(tgt_labels) == len(prd_labels), \
			f"[Error] Different number of target and predicted labels:\n" \
			f"{tgt_labels} (N={len(tgt_labels)})\n" \
			f"{prd_labels} (N={len(prd_labels)})"

		num_correct += sum([1 for tl, pl in zip(tgt_labels, prd_labels) if tl == pl])
		num_total += len(tgt_labels)

	accuracy = num_correct/num_total
	print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
	main()
