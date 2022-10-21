# Spectral Probing

> Follow the rainbow 🌈

This archive contains implementations of the methods from "**Spectral Probing**" ([Müller-Eberstein, van der Goot and Plank, 2022c](https://personads.me/x/emnlp-2022-paper); EMNLP 2022);.

![Schematic of spectral probing. A single row from a sequence of embeddings is represented as a wave built from each cell's value (low to high). The wave is decomposed into its composite frequencies using DCT. The spectral probe gamma continuously weights each frequency. The filtered frequencies are recomposed into the output embeddings using IDCT.](spectral-probing.png)

It enables probing for task-specific information using unfiltered, manually filtered and automatically filtered representations using the following filter definitions:

* `nofilter()`: retains the original embeddings.
* `eqalloc(mf, nb, bi)`: divides the `max_freqs` frequencies into `num_bands` equally allocated bands and allows only the waves of the band at `band_idx` to pass.
* `band(mf, si, ei)`: allows only the frequencies between `start_idx` and `end_idx` out of `max_freqs` to pass.
* `auto(mf)`: initializes a continuous learnable filter (i.e., spectral probe) with `max_freqs` active frequencies which are tuned in parallel with the task-specific head.

The toolkit is relatively flexible and can be applied to any token or sequence-level classification task once it has been converted into a unified CSV-format.

After installing the required packages, and downloading external datasets, the experiments can be re-run using the `run.sh` scripts in the appropriate task sub-directories. Please see the instructions below for details.

## Installation

This repository uses Python 3.6+ and the associated packages listed in the `requirements.txt` (a virtual environment is recommended):

```bash
(venv) $ pip install -r requirements.txt
```

## Data

Token-level tasks should be formatted as, with the number of space-separated tokens exactly matching the number of labels:

```
"text","label"
"token0 token1 token2","label0 label1 label2"
...
```

Sequence-level tasks should be formatted with a single label per sequence in the label column:

```
"text","label"
"The first sequence.","label0"
...
```

For sequence-level tasks involving two inputs (e.g., natural language inference), two text columns and one label column should be provided to ensure the correct segment encoding:

```
"text0","text1","label"
"The first sequence.","The second sequence.",label0"
...
```

## Experiments

Running an experiment involves training a classification head together with a specified filter (see `classify.py --help` for details):

```bash
(venv) $ python classify.py \
      data/train.csv data/dev.csv --repeat_labels \
      "encoder" --embedding_caching \
      "filter()" \
      "classifier" \
      exp/task/ \
      --random_seed 42
```

To run inference only, add the `--prediction` flag to the command above.

In order to compute the evaluation metrics for a given prediction, there are both a token-level evaluation utility as well as a sentence-level utility:

```bash
(venv) $ python tasks/eval/tokens.py data/target.csv exp/task/prediction.csv -t "tokenizer" 
(venv) $ python tasks/eval/sentences.py data/target.csv exp/task/prediction.csv
```

### EMNLP 2022

The following lists the experiments included in this repository, in addition to whether datasets are obtained automatically or require manual setup (e.g., due to licensing):

* **20 Newsgroups (Lang, 1995)** located in `tasks/20news/` requires a spearate download of the original data (please use the version in `20news-bydate.tar.gz`). Dataset conversion and experiments can be run using `tasks/20news/run.sh`.
* **Multilingual Amazon Reviews (Keung et al., 2020)** located in `tasks/amazon/` includes dataset downloading, conversion and experiments in `tasks/amazon/run.sh`.
* **JSNLI (Yoshikoshi et al., 2020)** located in `tasks/jsnli/` requires a spearate download of the original data. Dataset conversion and experiments can be run using `tasks/jsnli/run.sh`.
* **MKQA (Longpre et al., 2021)** located in `tasks/mkqa/` includes dataset downloading, conversion and experiments in `tasks/mkqa/run.sh`.
* **Penn Treebank (Marcus et al., 1993)** located in `tasks/ptb/` requires a spearate download of the original data. Dataset conversion and experiments can be run using `tasks/ptb/run.sh`.
* **Universal Dependencies (Zeman et al., 1993)** located in `tasks/ud-syntax/` requires a spearate download of the original data. Dataset conversion and experiments can be run using `tasks/ud-syntax/run.sh`.
* **WikiANN (Pan et al., 2017)** located in `tasks/wikiann/` includes dataset downloading, conversion and experiments in `tasks/wikiann/run.sh`.
* **XNLI (Conneau et al., 2018)** located in `tasks/xnli/` includes dataset downloading, conversion and experiments in `tasks/xnli/run.sh`.

Each task sub-directory contains a dataset conversion script (`convert.py`) and a `run.sh` script which calls the conversion, training and evaluation scripts with the appropriate parameters. By default, these scripts use `~/data/` and `~/exp/spectral/` to store data and experiments respectively. Please make sure to update them to your machine (if necessary).
