# Spectral Probing

This archive contains anonymized implementations of the methods from the EMNLP 2022 submission "**Spectral Probing**".

After installing the required packages, and downloading external datasets, the experiments can be re-run using the `run.sh` scripts in the appropriate task sub-directories. Please see the instructions below for details.

## Installation

This repository uses Python 3.6+ and the associated packages listed in the `requirements.txt` (a virtual environment is recommended):

```bash
(venv) $ pip install -r requirements.txt
```

## Experiments

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
