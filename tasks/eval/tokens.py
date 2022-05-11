#!/usr/bin/python3

import argparse, os, sys

import transformers

from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.datasets import LabelledDataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Evaluate Token-level Classification')
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
		# convert token IDs to pieces
		pieces = tokenizer.convert_ids_to_tokens(tok_text['input_ids'])
		for tidx in range(sum(tok_text['attention_mask'])):
			if tok_text['special_tokens_mask'][tidx] == 1: continue

			# check for start of new token
			if tok_text['offset_mapping'][tidx][0] == 0:
				# check for incorrect offset mapping in SentencePiece tokenizers (e.g. XLM-R)
				# example: ',' -> '▁', ',' with [0, 1], [0, 1] which increment the label cursor prematurely
				# https://github.com/huggingface/transformers/issues/9637
				if (tidx > 0) and (pieces[tidx - 1] != '▁'):
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
		tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
		print(f"Loaded '{args.tokenizer}' ({tokenizer.__class__.__name__}) for label repetition.")

	num_correct, num_total = 0, 0
	targets, predictions = [], []
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

		targets += tgt_labels
		predictions += prd_labels

	accuracy = num_correct/num_total
	print(f"Accuracy: {accuracy * 100:.2f}%")

	f1_macro = f1_score(targets, predictions, average='macro')
	print(f"F1 (macro): {f1_macro * 100:.2f}%")

	f1_micro = f1_score(targets, predictions, average='micro')
	print(f"F1 (micro): {f1_micro * 100:.2f}%")


if __name__ == '__main__':
	main()
