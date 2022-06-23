import argparse, csv, os

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='XNLI - Dataset Conversion')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpus')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	print("Loading XNLI dataset from HuggingFace...")
	languages = ['de', 'en', 'es', 'fr', 'zh']
	split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}
	label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

	for language in languages:
		xnli = load_dataset('xnli', language)
		for split_name, split in xnli.items():
			# gather relevant fields for all instances
			print(f"Converting '{language}' ({split_name}) data...")
			premises, hypotheses, labels = [], [], []
			for idx, instance in enumerate(split):
				premises.append(instance['premise'])
				hypotheses.append(instance['hypothesis'])
				labels.append(label_map[instance['label']])

			assert len(premises) == len(hypotheses) == len(labels),\
				f"[Error] Unequal number of premises (N={len(premises)}), hypotheses (N={len(hypotheses)}) " \
				f"and labels (N={len(labels)}."

			# write to CSV
			split_path = os.path.join(args.output_path, f'{language}-{split_map[split_name]}.csv')
			with open(split_path, 'w', encoding='utf8', newline='') as output_file:
				csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
				csv_writer.writerow(['text0', 'text1', 'label'])
				csv_writer.writerows(zip(premises, hypotheses, labels))
			print(f"Saved {len(labels)} instances to '{split_path}'.")


if __name__ == '__main__':
	main()
