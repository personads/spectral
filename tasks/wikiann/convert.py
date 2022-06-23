import argparse, csv, json, os

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='WikiANN - Dataset Conversion')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpus')
	return arg_parser.parse_args()


def save_dataset(path, texts, labels):
	with open(path, 'w', encoding='utf8', newline='') as output_file:
		csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
		csv_writer.writerow(['text', 'label'])
		csv_writer.writerows(zip(texts, labels))
	print(f"Saved {len(texts)} instances to '{path}'.")


def main():
	args = parse_arguments()

	print("Loading WikiANN (Rahimi et al., 2019) dataset from HuggingFace...")
	languages = ['de', 'en', 'es', 'fr', 'ja', 'zh']
	split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}
	label_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

	for language in languages:
		wikiann = load_dataset('wikiann', language)
		for split_name, split in wikiann.items():
			# gather relevant fields for all instances
			print(f"Converting '{language}' ({split_name}) data...")
			texts, labels = [], []
			for idx, instance in enumerate(split):
				texts.append(' '.join(instance['tokens']))
				labels.append(' '.join([label_map[lidx] for lidx in instance['ner_tags']]))

			# write to CSV
			split_path = os.path.join(args.output_path, f'{language}-{split_map[split_name]}.csv')
			save_dataset(split_path, texts, labels)


if __name__ == '__main__':
	main()
