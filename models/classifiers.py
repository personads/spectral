import torch
import torch.nn as nn

from models.encoders import *
from models.losses import *

#
# Base Classifier
#


class EmbeddingClassifier(nn.Module):
	def __init__(self, emb_model, lbl_model, classes, emb_pooling=None):
		super().__init__()
		# internal models
		self._emb = emb_model
		self._emb_pooling = emb_pooling
		self._lbl = lbl_model
		# internal variables
		self._classes = classes
		# move model to GPU if available
		if torch.cuda.is_available():
			self.to(torch.device('cuda'))

	def __repr__(self):
		return f'''<{self.__class__.__name__}:
	emb_model = {self._emb},
	emb_pooling = {self._emb_pooling},
	num_classes = {len(self._classes)}
>'''

	def get_trainable_parameters(self):
		return list(self._emb.get_trainable_parameters()) + list(self._lbl.parameters())

	def get_savable_objects(self):
		objects = self._emb.get_savable_objects()
		objects['classifier'] = self._lbl
		objects['classes'] = self._classes
		return objects

	def save(self, path):
		torch.save(self.get_savable_objects(), path)

	@staticmethod
	def load(path, emb_model, emb_pooling=None):
		objects = torch.load(path)
		classes = objects['classes']
		lbl_model = objects['classifier']
		# instantiate class using pre-trained label model and fixed encoder
		return EmbeddingClassifier(
			emb_model=emb_model, lbl_model=lbl_model, classes=classes, emb_pooling=emb_pooling
		)

	def forward(self, sentences):
		# embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
		emb_tokens, att_tokens = self._emb(sentences)

		# pool token embeddings into sentence representations
		if self._emb_pooling is not None:
			# prepare sentence embedding tensor (batch_size, 1, emb_dim)
			emb_sentences = torch.zeros((emb_tokens.shape[0], 1, emb_tokens.shape[2]), device=emb_tokens.device)
			# iterate over sentences and pool relevant tokens
			for sidx in range(emb_tokens.shape[0]):
				emb_sentences[sidx, 0, :] = self._emb_pooling(emb_tokens[sidx, :torch.sum(att_tokens[sidx]), :])
			# set embedding attention mask to cover each sentence embedding
			att_sentences = torch.ones((att_tokens.shape[0], 1), dtype=torch.bool)
		# when not pooling, the embedded sentences retain an embedding per token
		else:
			emb_sentences = emb_tokens
			att_sentences = att_tokens

		# logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
		logits = torch.ones(
			(att_sentences.shape[0], att_sentences.shape[1], len(self._classes)),
			device=emb_sentences.device
		) * float('-inf')
		# get token embeddings of all sentences (total_tokens, emb_dim)
		emb_tokens = emb_sentences[att_sentences, :]
		# pass through classifier
		flat_logits = self._lbl(emb_tokens)  # (num_words, num_labels)
		logits[att_sentences, :] = flat_logits  # (batch_size, max_len, num_labels)

		labels = self.get_labels(logits.detach())

		results = {
			'labels': labels,
			'logits': logits,
			'flat_logits': flat_logits
		}

		return results

	def get_labels(self, logits):
		# get predicted labels with maximum probability (padding should have -inf)
		labels = torch.argmax(logits, dim=-1)  # (batch_size, max_len)
		# add -1 padding label for -inf logits
		labels[(logits[:, :, 0] == float('-inf'))] = -1

		return labels


#
# Classifiers
#


class LinearClassifier(EmbeddingClassifier):
	def __init__(self, emb_model, classes, bias=True, emb_pooling=None):
		# instantiate linear classifier without bias
		lbl_model = nn.Linear(emb_model.emb_dim, len(classes), bias=bias)

		super().__init__(
			emb_model=emb_model, lbl_model=lbl_model, classes=classes,
			emb_pooling=emb_pooling
		)


#
# Helper Functions
#

def load_classifier(identifier):
	if identifier == 'linear':
		return LinearClassifier, LabelLoss
	else:
		raise ValueError(f"[Error] Unknown classifier specification '{identifier}'.")