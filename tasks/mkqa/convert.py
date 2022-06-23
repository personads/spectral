import argparse, csv, os, sys

import numpy as np

from collections import defaultdict
from datasets import load_dataset


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='MKQA - Dataset Conversion')
	arg_parser.add_argument('output_path', help='prefix for CSV output corpus')
	arg_parser.add_argument(
		'-ni', '--num_incorrect', type=int, default=1,
		help='proportion of incorrect answers to generate per instance (default: 1)')
	arg_parser.add_argument(
		'-vp', '--validation_proportion', type=float, default=.2,
		help='proportion of training data to use as validation data (default: .2)')
	arg_parser.add_argument(
		'-rs', '--random_seed', type=int, help='seed for probabilistic components (default: None)')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# set random seed
	np.random.seed(args.random_seed)

	print("Loading MKQA dataset from HuggingFace...")
	mkqa = load_dataset('mkqa')['train']  # MKQA only has a train-split
	languages = ['de', 'en', 'es', 'fr', 'ja', 'zh_cn']

	# gather answer types
	answer_types = defaultdict(list)
	for idx, instance in enumerate(mkqa):
		answer_types[instance['answers']['en'][0]['type']].append(idx)
	print(f"Gathered answer type distribution:  {', '.join([f'{t}: {len(idcs)}' for t, idcs in sorted(answer_types.items())])}.")

	# generate incorrect answers (same type, different text, consistent across langs)
	incorrect_indices = {}  # {idx: [inc_idx0, inc_idx1], ...}
	for idx, instance in enumerate(mkqa):
		sys.stdout.write(f'\r[{idx + 1}/{len(mkqa)}] Generating incorrect answers...')
		sys.stdout.flush()
		correct_answer = instance['answers']['en'][0]['text']
		correct_type = instance['answers']['en'][0]['type']
		# skip unanswerable questions
		if correct_answer is None:
			continue
		# check if there are enough alternatives to sample from
		assert args.num_incorrect < len(answer_types[correct_type]), \
			f"[Error] {args.num_incorrect} incorrect answers requested, " \
			f"but there are only {len(answer_types[correct_type])} instances of type {correct_type}."
		# sample as many incorrect answers as requested
		for _ in range(args.num_incorrect):
			cur_incorrect_indices = []
			# sample answers of the same type until their text does not match the correct one
			while True:
				incorrect_idx = int(np.random.choice(answer_types[correct_type], 1)[0])
				incorrect_candidate = mkqa[incorrect_idx]['answers']['en'][0]['text']
				if correct_answer != incorrect_candidate:
					cur_incorrect_indices.append(incorrect_idx)
					break
			incorrect_indices[idx] = cur_incorrect_indices
	print()
	print(f"Generated {args.num_incorrect} incorrect answer(s) per question using seed {args.random_seed}.")

	# generate splits
	shuffled_indices = np.arange(len(mkqa))
	np.random.shuffle(shuffled_indices)
	# split off validation data from training
	split_idx = round(len(mkqa) * args.validation_proportion)
	splits = {'train': sorted(shuffled_indices[split_idx:].tolist()), 'dev': sorted(shuffled_indices[:split_idx].tolist())}
	print(f"Created splits: {', '.join([f'{s}: {len(idcs)}' for s, idcs in splits.items()])} using seed {args.random_seed}.")

	for language in languages:
		for split_name, split_indices in splits.items():
			print(f"Converting '{language}' ({split_name}) data...")
			questions, answers, labels = [], [], []
			for idx in split_indices:
				instance = mkqa[idx]
				# skip unanswerable questions
				if instance['answers'][language][0]['text'] is None:
					continue
				# add correct answer
				questions.append(instance['queries'][language])
				answers.append(instance['answers'][language][0]['text'])
				labels.append('1')
				# add incorrect answers
				for inc_idx in incorrect_indices[idx]:
					questions.append(instance['queries'][language])
					answers.append(mkqa[inc_idx]['answers'][language][0]['text'])
					labels.append('0')
			assert len(questions) == len(answers) == len(labels), \
				f"[Error] Unequal number of questions (N={len(questions)}), answers (N={len(answers)}) " \
				f"and labels (N={len(labels)}."

			# write to CSV
			lang_alt = language.split('_')[0]  # convert 'aa_bb' to 'aa'
			split_path = os.path.join(args.output_path, f'{lang_alt}-{split_name}.csv')
			with open(split_path, 'w', encoding='utf8', newline='') as output_file:
				csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
				csv_writer.writerow(['text0', 'text1', 'label'])
				csv_writer.writerows(zip(questions, answers, labels))
			print(f"Saved {len(labels)} instances to '{split_path}'.")


if __name__ == '__main__':
	main()
