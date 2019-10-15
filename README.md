# phonotactic-complexity

This repository contains code for analysing phonotactics.

It is a study about languages phonotactics and how it relates to other language features, such as word length.

## Install Dependencies

Create a conda environment with
```bash
$ source config/conda.sh
```
And install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).

## Parse data

First, download NorthEuraLex data from [this link](http://www.northeuralex.org/) and put it in the `datasets/northeuralex` folder.
Then, parse it using the following command:

```bash
$ python data_layer/parse.py --data northeuralex
```

## Train models

### Train base models

You can train the base models (without shared embeddings) with the commands:

```bash
$ python learn_layer/train_base.py --model <model> [--opt]
$ python learn_layer/train_base_bayes.py --model <model>
$ python learn_layer/train_base_cv.py --model <model> [--opt]
```

Different commands are:
* `train_base`: Trains model with default data split;
* `train_base_bayes`: Trains model using bayesing optimization and default data split;
* `train_base_cv`: Trains cross validated models.

Model can be:
* lstm: LSTM with default one hot embeddings
* phoible: LSTM with phoible embeddings
* phoible-lookup: LSTM with both one hot and phoible embeddings


And `--opt` is an optional parameter that tells the script to use bayes optimized hyper-parameters. It can only be used after training model with `train_base_bayes`.

### Train shared models

You can train models with shared embeddings using the commands:
```bash
$ python learn_layer/train_shared.py --model <model> [--opt]
$ python learn_layer/train_shared_bayes.py --model <model>
$ python learn_layer/train_shared_cv.py --model <model> [--opt]
```

Model can be:
* shared-lstm: LSTM with shared one hot embeddings
* shared-phoible: LSTM with shared phoible embeddings
* shared-phoible-lookup: LSTM with both one hot and phoible shared embeddings


### Train ngram models

You can train ngram models with the following commands:
```bash
$ python learn_layer/train_ngram.py --model ngram
$ python learn_layer/train_unigram.py --model unigram

$ python learn_layer/train_ngram_cv.py --model ngram
$ python learn_layer/train_unigram_cv.py --model ngram
```

Model can be:
* ngram: ngram model by default is a trigram
* unigram: Unigram model


### Train models on artificial data

You can train models on aritificial data using the commands:
```bash
$ python learn_layer/train_artificial.py --artificial-type <artificial-type>
$ python learn_layer/train_artificial_bayes.py --artificial-type <artificial-type>
$ python learn_layer/train_artificial_cv.py --artificial-type <artificial-type>
$ python learn_layer/train_artificial_ngram.py --model ngram --artificial-type <artificial-type>
```

Artificial type can be:
* harmony: Artificial data with vowel harmony removed;
* devoicing: Artificial data with final obstruent devoicing removed.


### Train all models

You can also call a script to train all models sequentially (it might take a while):

```bash
$ source learn_layer/train_multi.sh
```

## Plot Results

Get compiled result data:

```bash
$ python analysis_layer/compile_results.py
$ python analysis_layer/get_lang_inventory.py
```

Plot all results with commands:
```bash
$ mkdir plot
$ python visualization_layer/plot_lstm.py
$ python visualization_layer/plot_full.py
$ python visualization_layer/plot_inventory.py
$ python visualization_layer/plot_kde.py
$ python visualization_layer/plot_artificial_scatter.py
```

## Extra Information

#### Dependencies

This project was tested with libraries:
```bash
numpy==1.15.4
pandas==0.24.1
scikit-learn==0.20.2
tqdm==4.31.1
matplotlib==2.0.2
seaborn==0.9.0
torch==1.0.1.post2
```

#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/phonotactic-complexity/issues).
