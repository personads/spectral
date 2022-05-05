#!/usr/bin/python3

import argparse, logging, os, sys

import numpy as np

from collections import defaultdict

# local imports
from utils.setup import *
from utils.datasets import LabelledDataset
from utils.training import classify_dataset
from models.encoders import *
from models.classifiers import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Classifier Training')
	# data setup
	arg_parser.add_argument('train_path', help='path to training data')
	arg_parser.add_argument('valid_path', help='path to validation data')
	arg_parser.add_argument(
		'-rl', '--repeat_labels', action='store_true', default=False,
		help='set flag to repeat sequence target label for each token in sequence (default: False)')
	arg_parser.add_argument(
		'-p', '--prediction', action='store_true', default=False,
		help='set flag to only perform prediction on the validation data without training (default: False)')

	# encoder setup
	arg_parser.add_argument('lm_name', help='embedding model identifier')
	arg_parser.add_argument('filter', help='type of frequency filter to apply')
	arg_parser.add_argument(
		'-ec', '--embedding_caching', action='store_true', default=False,
		help='flag to activate RAM embedding caching (default: False)')
	arg_parser.add_argument(
		'-et', '--embedding_tuning', action='store_true', default=False,
		help='set flag to tune the full model including embeddings (default: False)')
	arg_parser.add_argument(
		'-ep', '--embedding_pooling', choices=['mean'], help='embedding pooling strategy (default: None)')

	# classifier setup
	arg_parser.add_argument('classifier', help='classifier identifier')

	# experiment setup
	arg_parser.add_argument('exp_path', help='path to experiment directory')
	arg_parser.add_argument(
		'-e', '--epochs', type=int, default=30,
		help='maximum number of epochs (default: 50)')
	arg_parser.add_argument(
		'-es', '--early_stop', type=int, default=1,
		help='maximum number of epochs without improvement (default: 1)')
	arg_parser.add_argument(
		'-bs', '--batch_size', type=int, default=32,
		help='maximum number of sentences per batch (default: 32)')
	arg_parser.add_argument(
		'-lr', '--learning_rate', type=float, default=1e-3,
		help='learning rate (default: 1e-3)')
	arg_parser.add_argument(
		'-dr', '--decay_rate', type=float, default=.5,
		help='learning rate decay (default: 0.5)')
	arg_parser.add_argument(
		'-rs', '--random_seed', type=int,
		help='seed for probabilistic components (default: None)')

	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# set up experiment directory and logging
	setup_experiment(args.exp_path, prediction=args.prediction)

	if args.prediction:
		logging.info("Running in prediction mode (no training).")
		model_path = os.path.join(args.exp_path, 'best.pt')
		if not os.path.exists(model_path):
			logging.error(f"[Error] No pre-trained model available at '{model_path}'. Exiting.")
			exit(1)

	# set random seeds
	if args.random_seed is not None:
		np.random.seed(args.random_seed)
		torch.random.manual_seed(args.random_seed)

	# set up data
	valid_data = LabelledDataset.from_path(args.valid_path)
	label_types = sorted(set(valid_data.get_label_types()))
	logging.info(f"Loaded {valid_data} (dev).")
	if args.train_path.lower() != 'none':
		train_data = LabelledDataset.from_path(args.train_path)
		logging.info(f"Loaded {train_data} (train).")
		# gather labels
		if set(train_data.get_label_types()) < set(valid_data.get_label_types()):
			logging.warning(f"[Warning] Validation data contains labels unseen in the training data.")
		label_types = sorted(set(train_data.get_label_types()) | set(valid_data.get_label_types()))

	# set up frequency filter
	frq_filter = setup_filter(args.filter)
	frq_tuning = args.filter.startswith('auto(')
	logging.info(f"Loaded frequency filter '{args.filter}'.")

	# load pooling strategy
	pooling_strategy = None if args.embedding_pooling is None else load_pooling_function(args.embedding_pooling)

	# load encoder
	encoder = PrismEncoder(
		lm_name=args.lm_name, frq_filter=frq_filter, frq_tuning=frq_tuning,
		emb_tuning=args.embedding_tuning, emb_pooling=pooling_strategy,
		cache=({} if args.embedding_caching else None))
	logging.info(f"Constructed {encoder}.")
	if args.prediction:
		encoder = PrismEncoder.load(
			model_path, frq_filter=frq_filter, frq_tuning=frq_tuning,
			emb_tuning=args.embedding_tuning, emb_pooling=pooling_strategy,
			cache=({} if args.embedding_caching else None)
		)
		logging.info(f"Loaded pre-trained encoder from '{model_path}'.")

	# load classifier and loss constructors based on identifier
	classifier_constructor, loss_constructor = load_classifier(args.classifier)

	# setup classifier
	classifier = classifier_constructor(
		emb_model=encoder, classes=label_types
	)
	logging.info(f"Constructed classifier:\n{classifier}")
	# load pre-trained classifier
	if args.prediction:
		classifier = classifier.load(
			model_path, emb_model=encoder
		)
		logging.info(f"Loaded pre-trained classifier from '{model_path}'.")

	# setup loss
	criterion = loss_constructor(label_types)
	logging.info(f"Using criterion {criterion}.")

	# main prediction call (when only predicting on validation data w/o training)
	if args.prediction:
		stats = classify_dataset(
			classifier, criterion, None, valid_data,
			args.batch_size, repeat_labels=args.repeat_labels, mode='eval', return_predictions=True
		)
		# convert label indices back to string labels
		idx_lbl_map = {idx: lbl for idx, lbl in enumerate(label_types)}
		pred_labels = [
			[idx_lbl_map[p] for p in preds]
			for preds in stats['predictions']
		]
		pred_data = LabelledDataset(valid_data._inputs, pred_labels)
		pred_path = os.path.join(args.exp_path, f'{os.path.splitext(os.path.basename(args.valid_path))[0]}-pred.csv')
		pred_data.save(pred_path)
		logging.info(f"Prediction completed with Acc: {np.mean(stats['accuracy']):.4f}, Loss: {np.mean(stats['loss']):.4f} (mean over batches).")
		logging.info(f"Saved results from {pred_data} to '{pred_path}'. Exiting.")
		exit()

	# setup optimizer and scheduler
	optimizer = torch.optim.Adam(params=classifier.get_trainable_parameters(), lr=args.learning_rate)
	logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.decay_rate, patience=0)
	logging.info(f"Scheduler {scheduler.__class__.__name__} reduces learning rate by {args.decay_rate} after 1 epoch without improvement.")

	# main training loop
	stats = defaultdict(list)
	for ep_idx in range(args.epochs):
		# iterate over training batches and update classifier weights
		ep_stats = classify_dataset(
			classifier, criterion, optimizer, train_data,
			args.batch_size, repeat_labels=args.repeat_labels, mode='train'
		)
		# print statistics
		logging.info(
			f"[Epoch {ep_idx + 1}/{args.epochs}] Train completed with "
			f"Acc: {np.mean(ep_stats['accuracy']):.4f}, Loss: {np.mean(ep_stats['loss']):.4f}"
		)

		# iterate over batches in dev split
		ep_stats = classify_dataset(
			classifier, criterion, None, valid_data,
			args.batch_size, repeat_labels=args.repeat_labels, mode='eval'
		)

		# store and print statistics
		for stat in ep_stats:
			stats[stat].append(np.mean(ep_stats[stat]))
		logging.info(
			f"[Epoch {ep_idx + 1}/{args.epochs}] Validation completed with "
			f"Acc: {stats['accuracy'][-1]:.4f}, Loss: {stats['loss'][-1]:.4f}"
		)
		cur_eval_loss = stats['loss'][-1]

		# save most recent model
		path = os.path.join(args.exp_path, 'newest.pt')
		classifier.save(path)
		logging.info(f"Saved model from epoch {ep_idx + 1} to '{path}'.")

		# save best model
		if cur_eval_loss <= min(stats['loss']):
			path = os.path.join(args.exp_path, 'best.pt')
			classifier.save(path)
			logging.info(f"Saved model with best loss {cur_eval_loss:.4f} to '{path}'.")

		# check for early stopping
		if (ep_idx - stats['loss'].index(min(stats['loss']))) > args.early_stop:
			logging.info(f"No improvement since {args.early_stop + 1} epochs ({min(stats['loss']):.4f} loss). Early stop.")
			break

	logging.info(f"Training completed after {ep_idx + 1} epochs.")


if __name__ == '__main__':
	main()
