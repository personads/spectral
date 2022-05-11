import csv
import sys

import numpy as np


class LabelledDataset:
    def __init__(self, inputs, labels):
        self._inputs = inputs  # List(List(Str)): [['t0', 't1', ...], ['t0', 't1', ...]] or List(Str): ['t0 t1 ... tN']
        self._labels = labels  # List(List(Str)): [['l0', 'l1', ...], ['l0', 'l1', ...]] or List(Str): ['l0', 'l1', ...]
        self._label_level = None

    def __len__(self):
        return len(list(self.get_flattened_labels()))

    def __repr__(self):
        return f'<LabelledDataset: {len(self._inputs)} inputs, {len(self)} labels ({self.get_label_level()}-level)>'

    def __getitem__(self, key):
        return self._inputs[key], self._labels[key]

    def get_input_count(self):
        return len(self._inputs)

    def get_flattened_labels(self):
        for cur_labels in self._labels:
            if type(cur_labels) is list:
                for cur_label in cur_labels:
                    yield cur_label
            else:
                yield cur_labels

    def get_label_types(self):
        label_types = set()
        for label in self.get_flattened_labels():
            label_types.add(label)
        return sorted(label_types)

    def get_label_level(self):
        level = self._label_level
        if level is not None:
            return self._label_level

        for idx in range(self.get_input_count()):
            if type(self._labels[idx]) is list:
                if level is None:
                    level = 'token'
                elif level == 'sequence':
                    level = 'multiple'
                    break
            else:
                if level is None:
                    level = 'sequence'
                elif level == 'token':
                    level = 'multiple'
                    break
        self._label_level = level

        return level

    def get_batches(self, batch_size):
        cursor = 0
        while cursor < len(self._inputs):
            # set up batch range
            start_idx = cursor
            end_idx = min(start_idx + batch_size, len(self._inputs))
            cursor = end_idx
            num_remaining = len(self._inputs) - cursor - 1
            # slice data
            inputs = self._inputs[start_idx:end_idx]
            labels = self._labels[start_idx:end_idx]
            # flatten sequential labels if necessary
            if self.get_label_level() == 'token':
                labels = [l for seq in labels for l in seq]
            # yield batch
            yield inputs, labels, num_remaining

    def get_shuffled_batches(self, batch_size):
        # start with list of all input indices
        remaining_idcs = list(range(len(self._inputs)))
        np.random.shuffle(remaining_idcs)

        # generate batches while indices remain
        while len(remaining_idcs) > 0:
            # pop-off relevant number of instances from pre-shuffled set of remaining indices
            batch_idcs = [remaining_idcs.pop() for _ in range(min(batch_size, len(remaining_idcs)))]

            # gather batch data
            inputs = [self._inputs[idx] for idx in batch_idcs]
            # flatten sequential labels if necessary
            if self.get_label_level() == 'token':
                labels = [l for idx in batch_idcs for l in self._labels[idx]]
            # one label per input does not require flattening
            else:
                labels = [self._labels[idx] for idx in batch_idcs]
            # yield batch + number of remaining instances
            yield inputs, labels, len(remaining_idcs)

    def repeat_batch_labels(self, inputs, labels, encoder):
        rep_labels = []

        if self.get_label_level() == 'token':
            tok_inputs = encoder.tokenize([s.split(' ') for s in inputs], tokenized=True)
        else:
            tok_inputs = encoder.tokenize(inputs)

        lbl_cursor = -1
        for sidx in range(len(inputs)):
            # convert token IDs to pieces
            pieces = encoder._tok.convert_ids_to_tokens(tok_inputs['input_ids'][sidx])
            for tidx in range(tok_inputs['attention_mask'][sidx].sum()):
                # skip special tokens
                if (not encoder._specials) and (tok_inputs['special_tokens_mask'][sidx, tidx]):
                    continue
                # repeat token labels across all sub-tokens
                if self.get_label_level() == 'token':
                    # check for start of new token
                    if tok_inputs['offset_mapping'][sidx, tidx, 0] == 0:
                        # check for incorrect offset mapping in SentencePiece tokenizers (e.g. XLM-R)
                        # example: ',' -> '▁', ',' with [0, 1], [0, 1] which increment the label cursor prematurely
                        # https://github.com/huggingface/transformers/issues/9637
                        if (tidx > 0) and (pieces[tidx - 1] != '▁'):
                            lbl_cursor += 1
                    rep_labels.append(labels[lbl_cursor])
                # repeat single sequence label across all sequence tokens
                else:
                    rep_labels.append(labels[sidx])

        return rep_labels

    def save(self, path):
        with open(path, 'w', encoding='utf8', newline='') as output_file:
            csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
            # construct header for single-/multi-sequence inputs + labels
            header = [f'text{i}' for i in range(len(self._inputs[0]))] if type(self._inputs[0]) is tuple else ['text']
            header.append('label')
            csv_writer.writerow(header)
            # iterate over instances
            for idx, text in enumerate(self._inputs):
                row = []
                # add one row each for multi-sequence inputs
                if type(text) is tuple:
                    row += list(text)
                # convert token-level labels back to a single string
                label = self._labels[idx]
                if type(label) is list:
                    label = ' '.join([str(l) for l in label])
                row.append(label)
                # write row
                csv_writer.writerow(row)

    @staticmethod
    def from_path(path):
        inputs, labels = [], []
        label_level = 'sequence'
        with open(path, 'r', encoding='utf8', newline='') as fp:
            csv.field_size_limit(sys.maxsize)
            csv_reader = csv.DictReader(fp)
            for row in csv_reader:
                # convert all previous labels to token-level when encountering the first token-level label set
                if (' ' in row['label']) and (label_level != 'token'):
                    labels = [[l] for l in labels]
                    label_level = 'token'
                # covert current label(s) into appropriate form
                if label_level == 'token':
                    label = row['label'].split(' ')
                else:
                    label = row['label']
                # check if text consists of multiple sequences
                if len(csv_reader.fieldnames) > 2:
                    text = tuple([row[f'text{i}'] for i in range(len(csv_reader.fieldnames) - 1)])
                # otherwise, simply retrieve the text field
                else:
                    text = row['text']
                # append inputs and labels to overall dataset
                inputs.append(text)
                labels.append(label)

        return LabelledDataset(inputs, labels)