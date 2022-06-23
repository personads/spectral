import argparse, csv, os

from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Amazon Multilingual Reviews - Dataset Conversion')
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

	print("Loading Amazon Multilingual Reviews dataset from HuggingFace...")
	languages = ['de', 'en', 'es', 'fr', 'ja', 'zh']
	split_map = {'train': 'train', 'validation': 'dev', 'test': 'test'}
	sentiment_map = {1: '-', 2: '-', 3: 'o', 4: '+', 5: '+'}

	for language in languages:
		amazon = load_dataset('amazon_reviews_multi', language)
		for split_name, split in amazon.items():
			# gather relevant fields for all instances
			print(f"Converting '{language}' ({split_name}) data...")
			texts, sentiments, topics = [], [], []
			for idx, instance in enumerate(split):
				texts.append(instance['review_body'])
				sentiments.append(sentiment_map[instance['stars']])
				topics.append(instance['product_category'])

			# write to CSV
			split_path = os.path.join(args.output_path, f'{language}-{split_map[split_name]}-sentiment.csv')
			save_dataset(split_path, texts, sentiments)
			split_path = os.path.join(args.output_path, f'{language}-{split_map[split_name]}-topic.csv')
			save_dataset(split_path, texts, topics)


if __name__ == '__main__':
	main()
