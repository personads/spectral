import argparse, csv, os, sys


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='JSNLI - Dataset Conversion')
	arg_parser.add_argument('input_path', help='path to JSNLI directory with data in TSV-format')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpus')
	return arg_parser.parse_args()


def load_tsv(split_path):
	premises, hypotheses, labels = [], [], []
	with open(split_path, 'r', encoding='utf8', newline='') as fp:
		csv.field_size_limit(sys.maxsize)
		csv_reader = csv.reader(fp, delimiter='\t')
		for row in csv_reader:
			premises.append(row[1])
			hypotheses.append(row[2])
			labels.append(row[0])
	return premises, hypotheses, labels


def main():
	args = parse_arguments()

	print(f"Loading JSNLI dataset from '{args.input_path}'...")
	split_paths = {'train': 'train_w_filtering.tsv', 'dev': 'dev.tsv'}

	for split, split_path in split_paths.items():
		# load TSV file
		premises, hypotheses, labels = load_tsv(os.path.join(args.input_path, split_path))
		print(f"Loaded {len(labels)} instances in {split}-split.")
		assert len(premises) == len(hypotheses) == len(labels),\
			f"[Error] Unequal number of premises (N={len(premises)}), hypotheses (N={len(hypotheses)}) " \
			f"and labels (N={len(labels)}."

		# write to CSV
		output_split_path = os.path.join(args.output_path, f'{split}.csv')
		with open(output_split_path, 'w', encoding='utf8', newline='') as output_file:
			csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
			csv_writer.writerow(['text0', 'text1', 'label'])
			csv_writer.writerows(zip(premises, hypotheses, labels))
		print(f"Saved {len(labels)} instances to '{output_split_path}'.")


if __name__ == '__main__':
	main()
