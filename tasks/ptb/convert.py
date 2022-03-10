#!/usr/bin/python3

import argparse, csv, os, re

from collections import defaultdict


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Penn Treebank - Dataset Conversion')
	arg_parser.add_argument('input_path', help='path to PTB directory')
	arg_parser.add_argument('output_path', help='output directory for CSV corpus')
	return arg_parser.parse_args()


def parse_to_pos(parse_line):
	text, labels = [], []
	text_replacements = {
		'-LRB-': '(', '-RRB-': ')', '-LCB-': '{', '-RCB-': '}',
		'``': '\u201C', "''": '\u201d'
	}
	for pos, lex in re.findall(r'\(([A-Z\-\$#`\',\.:]+) ([^\(\)]+)\)', parse_line):
		text.append(lex if lex not in text_replacements else text_replacements[lex])
		labels.append(pos)
	return text, labels


def main():
	args = parse_arguments()

	# initialize output lines
	split_data = defaultdict(list)

	# iterate over split files
	for file_path in os.listdir(args.input_path):
		file_path = os.path.join(args.input_path, file_path)
		split_match = re.search(r'.*(train|dev|test)\.ptb$', file_path)
		if split_match is None: continue
		split_name = split_match[1]

		# read split file
		with open(file_path, 'r', encoding='ascii') as fop:
			for line in fop:
				text, labels = parse_to_pos(line)
				split_data[split_name].append([' '.join(text), ' '.join(labels)])

	# write splits to files
	for split, lines in split_data.items():
		split_path = os.path.join(args.output_path, f'{split}.csv')
		with open(split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text', 'label'])
			csv_writer.writerows(lines)
		print(f"Saved {split}-split with {len(lines)} items to '{split_path}'.")


if __name__ == '__main__':
	main()
