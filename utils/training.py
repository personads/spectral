import sys

import torch
import numpy as np

from collections import defaultdict


def classify_dataset(
		classifier, criterion, optimizer,
		dataset, batch_size, mode='train',
		repeat_labels=False, return_predictions=False):
	stats = defaultdict(list)

	# set model to training mode
	if mode == 'train':
		classifier.train()
		batch_generator = dataset.get_shuffled_batches
	# set model to eval mode
	elif mode == 'eval':
		classifier.eval()
		batch_generator = dataset.get_batches

	# iterate over batches
	for bidx, batch_data in enumerate(batch_generator(batch_size)):
		# set up batch data
		sentences, labels, num_remaining = batch_data
		# repeat labels
		if repeat_labels:
			labels = dataset.repeat_batch_labels(sentences, labels, classifier._emb)

		# when training, perform both forward and backward pass
		if mode == 'train':
			# zero out previous gradients
			optimizer.zero_grad()

			# forward pass
			predictions = classifier(sentences)

			# propagate loss
			loss = criterion(predictions['logits'], labels)
			loss.backward()
			optimizer.step()

		# when evaluating, perform forward pass without gradients
		elif mode == 'eval':
			with torch.no_grad():
				# forward pass
				predictions = classifier(sentences)
				# calculate loss
				loss = criterion(predictions['logits'], labels)

		# calculate accuracy
		accuracy = criterion.get_accuracy(predictions['logits'].detach(), labels)

		# store statistics
		stats['loss'].append(float(loss.detach()))
		stats['accuracy'].append(float(accuracy))

		# store predictions
		if return_predictions:
			# iterate over inputs items
			for sidx in range(predictions['labels'].shape[0]):
				# append non-padding predictions as list
				predicted_labels = predictions['labels'][sidx]
				stats['predictions'].append(predicted_labels[predicted_labels != -1].tolist())

		# print batch statistics
		pct_complete = (1 - (num_remaining / len(dataset._inputs))) * 100
		sys.stdout.write(
			f"\r[{mode.capitalize()} | Batch {bidx + 1} | {pct_complete:.2f}%] "
			f"Acc: {np.mean(stats['accuracy']):.4f}, Loss: {np.mean(stats['loss']):.4f}"
		)
		sys.stdout.flush()

	# clear line
	print("\r", end='')

	return stats
