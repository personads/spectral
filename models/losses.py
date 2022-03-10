import torch
import torch.nn as nn

#
# Loss Functions
#


class LabelLoss(nn.Module):
	def __init__(self, classes):
		super().__init__()
		self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1)
		self._classes = classes
		self._class2id = {c:i for i, c in enumerate(self._classes)}

	def __repr__(self):
		return \
			f'<{self.__class__.__name__}: loss=XEnt, num_classes={len(self._classes)}>'

	def _gen_target_labels(self, logits, targets):
		target_labels = torch.tensor(
			[self._class2id[c] for c in targets], dtype=torch.long,
			device=logits.device
		)  # (num_targets,)

		return target_labels

	def forward(self, logits, targets):
		# gather target labels
		target_labels = self._gen_target_labels(logits, targets)
		# flatten logits
		flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]  # (sum_over_seq_lens,)

		return self._xe_loss(flat_logits, target_labels)

	def get_accuracy(self, logits, targets):
		# gather target labels
		target_labels = self._gen_target_labels(logits, targets)
		# get labels from logits
		flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]  # (sum_over_seq_lens,)
		labels = torch.argmax(flat_logits, dim=-1)

		# compute label accuracy
		num_label_matches = torch.sum(labels == target_labels)
		accuracy = float(num_label_matches / labels.shape[0])

		return accuracy