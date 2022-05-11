import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class PrismEncoder(nn.Module):
	def __init__(
			self, lm_name, frq_filter,
			frq_tuning=False, emb_tuning=False,
			emb_pooling=None, specials=False, cache=None):
		super().__init__()
		# load transformer
		transformers.logging.set_verbosity_error()
		self._tok = transformers.AutoTokenizer.from_pretrained(lm_name, use_fast=True)
		self._lm = transformers.AutoModel.from_pretrained(lm_name, return_dict=True)
		self._emb_tuning = emb_tuning
		# load cache
		self._cache = cache  # {hash: torch.tensor (num_layers, sen_len, emb_dim)}
		# internal variables
		self._lm_name = lm_name
		self._specials = specials
		self._emb_pooling = emb_pooling
		# frequency filter parameter
		self._frq_tuning = frq_tuning
		self._frq_filter = None if frq_filter is None else nn.Parameter(frq_filter)
		# public variables
		self.emb_dim = self._lm.config.hidden_size
		self.num_layers = self._lm.config.num_hidden_layers

	def __repr__(self):
		return \
			f'<{self.__class__.__name__}: {self._lm.__class__.__name__} ("{self._lm_name}")' \
			f', {"tunable" if self._emb_tuning else "static"} embeddings' \
			f', {"no" if self._emb_pooling is None else "subword"} pooling' \
			f', {f"{torch.sum(self._frq_filter != 0)}/{self._frq_filter.shape[0]}" if self._frq_filter is not None else "all"} freqs active' \
			f', {"tunable" if self._frq_tuning else "static"} filter' \
			f'{f", with cache (size={len(self._cache)})" if self._cache is not None else ""}>'

	def get_savable_objects(self):
		objects = {}
		objects['language_model_name'] = self._lm_name
		objects['embedding_pooling'] = self._emb_pooling
		if self._emb_tuning:
			objects['language_model'] = self._lm
		if self._frq_tuning:
			objects['frequency_filter'] = self._frq_filter
		return objects

	def get_trainable_parameters(self):
		parameters = []
		if self._emb_tuning:
			parameters += list(self._lm.parameters())
		if self._frq_tuning:
			parameters.append(self._frq_filter)
		return parameters

	def save(self, path):
		torch.save(self.get_savable_objects(), path)

	@staticmethod
	def load(path, frq_filter=None, frq_tuning=False, emb_tuning=False, emb_pooling=None, specials=False, cache=None):
		objects = torch.load(path)
		# load necessary components
		lm_name = objects['language_model_name']
		emb_pooling = objects.get('embedding_pooling', emb_pooling)
		frq_filter = objects.get('frequency_filter', frq_filter)
		# construct model
		encoder = PrismEncoder(
			lm_name=lm_name, frq_filter=frq_filter, frq_tuning=frq_tuning, emb_tuning=emb_tuning,
			emb_pooling=emb_pooling, specials=specials, cache=cache
		)
		# load LM (if available)
		encoder._lm = objects.get('language_model', encoder._lm)
		return encoder

	def train(self, mode=True):
		super().train(mode)
		# disable LM training (incl. dropout)
		if not self._emb_tuning:
			self._lm.eval()
		# disable frequency filter training
		if (self._frq_filter is not None) and (not self._frq_tuning):
			self._frq_filter.requires_grad_(False)
		return self

	def forward(self, sentences):
		# embed sentences (standard last-layer encoding)
		if self._emb_tuning:
			emb_tokens, att_tokens = self.embed(sentences)
		else:
			with torch.no_grad():
				emb_tokens, att_tokens = self.embed(sentences)

		# apply filter if set
		if self._frq_filter is not None:
			# apply discrete cosine transform
			if self._frq_tuning:
				emb_tokens = self.filter(emb_tokens, att_tokens)
			else:
				with torch.no_grad():
					emb_tokens = self.filter(emb_tokens, att_tokens)

		# pool token embeddings
		if self._emb_pooling is not None:
			# prepare sentence embedding tensor (batch_size, 1, emb_dim)
			emb_pooled = torch.zeros((emb_tokens.shape[0], 1, emb_tokens.shape[2]), device=emb_tokens.device)
			# iterate over sentences and pool relevant tokens
			for sidx in range(emb_tokens.shape[0]):
				emb_pooled[sidx, 0, :] = self._emb_pooling(emb_tokens[sidx, :torch.sum(att_tokens[sidx]), :])
			emb_tokens = emb_pooled
			# set embedding attention mask to cover each sentence embedding
			att_tokens = torch.ones((att_tokens.shape[0], 1), dtype=torch.bool)

		return emb_tokens, att_tokens

	def embed(self, sentences):
		# try retrieving embeddings from cache
		emb_cache = self.retrieve(sentences)
		if emb_cache is not None:
			emb_tokens, att_tokens = emb_cache
			return emb_tokens, att_tokens

		# compute embeddings if not in cache
		tok_sentences = self.tokenize(sentences)
		model_inputs = {
			k: tok_sentences[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']
			if k in tok_sentences
		}

		# perform embedding forward pass
		model_outputs = self._lm(**model_inputs)
		hidden_states = model_outputs.last_hidden_state  # (batch_size, max_len, hidden_dim)
		emb_tokens, att_tokens = hidden_states, tok_sentences['attention_mask'].bool()

		# remove special tokens
		if (not self._specials) and (self._emb_pooling is None):
			# mask any non-padding token that is special
			non_special_mask = ~tok_sentences['special_tokens_mask'].bool() & att_tokens  # (batch_size, max_len)
			# new maximum length without special tokens
			non_special_max_len = torch.max(torch.sum(non_special_mask, -1))
			non_special_embeddings = torch.zeros_like(emb_tokens)[:, :non_special_max_len, :]  # (batch_size, new_max_len, hidden_dim)
			non_special_attentions = torch.zeros_like(att_tokens)[:, :non_special_max_len]  # (batch_size, new_max_len)
			# iterate over each sequence due to variable length
			for sidx in range(emb_tokens.shape[0]):
				# current sequence length up to which values should be filled
				seq_len = torch.sum(non_special_mask[sidx])
				# store non-special token embeddings up to new sequence length (rest is zeros)
				non_special_embeddings[sidx, :seq_len] = emb_tokens[sidx, non_special_mask[sidx], :]
				non_special_attentions[sidx, :seq_len] = att_tokens[sidx, non_special_mask[sidx]]
			# keep non-special embeddings only
			emb_tokens = non_special_embeddings
			att_tokens = non_special_attentions

		# store embeddings in cache (if cache is enabled)
		if self._cache is not None:
			self.cache(sentences, emb_tokens, att_tokens)

		return emb_tokens, att_tokens

	def tokenize(self, sentences, tokenized=False):
		# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
		# check for multi-sentence inputs
		if (len(sentences) > 0) and (type(sentences[0]) is tuple):
			sentences1, sentences2 = [s[0] for s in sentences], [s[1] for s in sentences]
			tok_sentences = self._tok(
				sentences1, sentences2,
				padding=True, truncation=True,
				return_tensors='pt', return_special_tokens_mask=True,
				return_offsets_mapping=True, is_split_into_words=tokenized
			)
		# use standard tokenization otherwise
		else:
			tok_sentences = self._tok(
				sentences,
				padding=True, truncation=True,
				return_tensors='pt', return_special_tokens_mask=True,
				return_offsets_mapping=True, is_split_into_words=tokenized
			)

		# move input to GPU (if available)
		if torch.cuda.is_available():
			tok_sentences = {k: v.to(torch.device('cuda')) for k, v in tok_sentences.items()}

		return tok_sentences

	def filter(self, embeddings, attentions):
		"""
		Decomposes embeddings into the specified frequency bands.

		:param embeddings: tensor of embeddings with shape (batch_size, max_length, hidden_dim)
		:param attentions: tensor of attention masks with shape (batch_size, max_length)
		:return: filtered embeddings with the same shape as the input
		"""
		emb_filtered = torch.zeros_like(embeddings)

		# apply discrete cosine transform
		for sidx in range(embeddings.shape[0]):
			# run DCT over length of sequence
			seq_len = torch.sum(attentions[sidx])
			seq_frequencies = self.dct(embeddings[sidx, :seq_len].T)  # (hidden_dim, seq_length)
			# pool filter to sequence length (filter_size, ) -> (1, seq_len)
			frq_filter = F.adaptive_avg_pool1d(self._frq_filter[None, :], int(seq_len))
			# frq_filter = F.adaptive_max_pool1d(self._frq_filter[None, :], int(seq_len))
			# normalize filter values to [0, 1] for learnable filter
			frq_filter = torch.sigmoid(frq_filter) if self._frq_tuning else frq_filter
			# element-wise multiplication of filter with each row (no expansion needed due to prior pooling)
			seq_filtered = seq_frequencies * frq_filter
			# perform inverse DCT
			emb_recomposed = self.idct(seq_filtered).T  # (seq_len, hidden_dim)
			emb_filtered[sidx, :seq_len, :] = emb_recomposed

		return emb_filtered

	@staticmethod
	def gen_band_filter(filter_size, start_idx, end_idx):
		assert start_idx < end_idx < filter_size, \
			f"[Error] Range {start_idx}-{end_idx} out of range for filter with {filter_size} frequencies."

		filter = torch.zeros(filter_size)
		filter[start_idx:end_idx+1] = 1

		return filter

	@staticmethod
	def gen_equal_allocation_filters(filter_size, num_bands):
		filters = []
		assert num_bands <= filter_size, \
			f"[Error] Cannot equally allocate more bands than there are frequencies ({num_bands} < {filter_size})."

		# get number of frequencies to equally allocate
		num_equal = filter_size // num_bands
		# get remainder of bands to unequally distribute (max. num_bands - 1)
		num_unequal = filter_size % num_bands
		# allocate remainder bands from the outside in (starting at band 0)
		extra_bands = set()
		cursor = 0
		for eidx in range(num_unequal):
			# if even iteration, add to left
			if eidx % 2 == 0:
				extra_bands.add(eidx + cursor)
			# if uneven, add to right
			else:
				extra_bands.add(num_bands - cursor - 1)
				cursor += 1

		# equally allocate cleanly divisible bands
		for bidx in range(num_bands):
			cur_filter = torch.zeros(filter_size)
			start_idx = bidx * num_equal
			end_idx = start_idx + num_equal + int(bidx in extra_bands)
			cur_filter[start_idx:end_idx] = 1
			filters.append(cur_filter)

		return filters

	@staticmethod
	def gen_auto_filter_initialization(filter_size):
		return torch.ones(filter_size)

	@staticmethod
	def dct(embeddings, norm=None):
		"""
		(Implemented as in https://github.com/zh217/torch-dct)
		Discrete Cosine Transform, Type II (a.k.a. the DCT)

		For the meaning of the parameter `norm`, see:
		https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

		:param x: the input signal
		:param norm: the normalization, None or 'ortho'
		:return: the DCT-II of the signal over the last dimension
		"""
		orig_shape = embeddings.shape
		N = embeddings.shape[-1]
		embeddings = embeddings.contiguous().view(-1, N)

		v = torch.cat([embeddings[:, ::2], embeddings[:, 1::2].flip([1])], dim=1)

		Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

		k = - torch.arange(N, dtype=embeddings.dtype, device=embeddings.device)[None, :] * np.pi / (2 * N)
		W_r = torch.cos(k)
		W_i = torch.sin(k)

		V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

		if norm == 'ortho':
			V[:, 0] /= np.sqrt(N) * 2
			V[:, 1:] /= np.sqrt(N / 2) * 2

		V = 2 * V.view(*orig_shape)

		return V

	@staticmethod
	def idct(frequencies, norm=None):
		"""
		(Implemented as in https://github.com/zh217/torch-dct)
		The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
		Our definition of idct is that idct(dct(x)) == x
		For the meaning of the parameter `norm`, see:
		https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
		:param X: the input signal
		:param norm: the normalization, None or 'ortho'
		:return: the inverse DCT-II of the signal over the last dimension
		"""

		orig_shape = frequencies.shape
		N = orig_shape[-1]

		X_v = frequencies.contiguous().view(-1, orig_shape[-1]) / 2

		if norm == 'ortho':
			X_v[:, 0] *= np.sqrt(N) * 2
			X_v[:, 1:] *= np.sqrt(N / 2) * 2

		k = torch.arange(orig_shape[-1], dtype=frequencies.dtype, device=frequencies.device)[None, :] * np.pi / (2 * N)
		W_r = torch.cos(k)
		W_i = torch.sin(k)

		V_t_r = X_v
		V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

		V_r = V_t_r * W_r - V_t_i * W_i
		V_i = V_t_r * W_i + V_t_i * W_r

		V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

		v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
		frequencies = v.new_zeros(v.shape)
		frequencies[:, ::2] += v[:, :N - (N // 2)]
		frequencies[:, 1::2] += v.flip([1])[:, :N // 2]

		return frequencies.view(*orig_shape)

	def retrieve(self, sentences):
		if self._cache is None:
			return None

		max_len = 0
		emb_tokens = torch.zeros((len(sentences), self._lm.config.max_position_embeddings, self.emb_dim))
		att_tokens = torch.zeros((len(sentences), self._lm.config.max_position_embeddings), dtype=torch.bool)

		# iterate over sentences
		for sidx, sentence in enumerate(sentences):
			# retrieve sentence embedding using string hash
			sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()
			# skip batch if not all sentences are in cache
			if sen_hash not in self._cache:
				return None

			# retrieve embeddings from cache
			emb_cached = self._cache[sen_hash]  # (sen_len, emb_dim)
			emb_tokens[sidx, :emb_cached.shape[0], :] = emb_cached
			att_tokens[sidx, :emb_cached.shape[0]] = True
			max_len = max_len if max_len > emb_cached.shape[0] else emb_cached.shape[0]

		# truncate batch to maximum length
		emb_tokens = emb_tokens[:, :max_len, :]
		att_tokens = att_tokens[:, :max_len]

		# move input to GPU (if available)
		if torch.cuda.is_available():
			emb_tokens = emb_tokens.to(torch.device('cuda'))
			att_tokens = att_tokens.to(torch.device('cuda'))

		return emb_tokens, att_tokens

	def cache(self, sentences, emb_tokens, att_tokens):
		# detach, duplicate and move embeddings to CPU
		emb_tokens = emb_tokens.detach().clone().cpu()

		# iterate over sentences
		for sidx, sentence in enumerate(sentences):
			# compute sentence hash
			sen_hash = hashlib.md5(' '.join(sentence).encode('utf-8')).hexdigest()
			# store cache entry
			self._cache[sen_hash] = emb_tokens[sidx, :torch.sum(att_tokens[sidx]), :]  # (sen_len, emb_dim)


#
# Pooling Functions
#


def get_mean_embedding(token_embeddings):
	return torch.mean(token_embeddings, dim=0)


#
# Helper Functions
#

def load_pooling_function(identifier):
	if identifier == 'mean':
		return get_mean_embedding
	else:
		raise ValueError(f"[Error] Unknown pooling specification '{identifier}'.")