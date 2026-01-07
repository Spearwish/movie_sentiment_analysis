# Movie Sentiment Analysis with Transformers

This project implements a custom **Transformer-based architecture** from scratch using PyTorch to perform binary
sentiment analysis (positive/negative) on the IMDb Large Movie Review Dataset.

This repository builds the entire training and evaluating pipeline from the ground up, including a custom **Byte Pair
Encoding (BPE)** tokenizer from **Andrej Karpathy** `minbpe` repository.

## Project Structure

```text
movie_sentiment_analysis/
├── assets/                  # stores trained BPE tokenizer models and vocab
├── datasets/                # data directory for train and test split
│   ├── train/               # directory intended for pos/ and neg/ subfolders
│   └── test/                # directory intended for pos/ and neg/ subfolders
├── minbpe/                  # custom BPE tokenizer implementation (copied from minbpe)
├── runs/                    # TensorBoard logging directory
├── weights/                 # best model weights for specific configuration
├── config.py                # global hyperparameters and configuration
├── dataset.py               # PyTorch dataset class and text normalization logic
├── evaluation.py            # script for evaluating the model on the test set
├── inference.py             # script for running inference on custom text inputs
├── model.py                 # custom Transformer architecture implementation
├── stats.py                 # dataset and tokenizer statistics analysis
├── train.py                 # main training loop with logging and validation
├── train_bpe_tokenizer.py   # script for training the BPE tokenizer on the train dataset
├── requirements.txt         # project dependencies
└── README.md
```

## Installation

1. Clone the repository:

```shell
git clone https://github.com/Spearwish/movie_sentiment_analysis
cd movie_sentiment_analysis
```

2. Install the dependencies:

```shell
pip install -r requirements.txt
```

## Data Setup

This project uses the **IMDb Large Movie Review Dataset** provided by Stanford.

### 1. Download the dataset

Download the dataset from the official Stanford website:

https://ai.stanford.edu/~amaas/data/sentiment/

---

### 2. Extract the dataset

After downloading, extract (unzip) the archive.

After extraction, you should have a folder named:

```
aclImdb_v1/
```

---

### 3. Copy `train/` and `test/` folders

Inside the `aclImdb_v1/aclImdb/` directory, you will find the following folders:

```
aclImdb_v1/aclImdb/
           ├── train/
           └── test/
```

Copy **both** the `train/` and `test/` folders into the following project directory:

```
movie_sentiment_analysis/datasets/
```

After copying, final directory structure should look like this:

```
movie_sentiment_analysis/
└── datasets/
    ├── train/
    └── test/
```

---

### 4. Verify setup

Make sure that both `train/` and `test/` folders contain `pos/` and `neg/` subfolders with movie review text files
inside. The dataset should include **25,000 reviews for training** and **25,000 reviews for testing**,
with each split containing **12,500 positive** and **12,500 negative** reviews.
If this structure is correct, the dataset is ready to use.

## Dataset Statistics

To gain a deeper understanding of the dataset, specifically regarding **sequence length** (context window)
and the impact of **text normalization**, refer to the precomputed statistics at the bottom of `stats.py`.
These metrics were derived from the 25,000 sample IMDb train set.

### Understanding the Metrics

```text
# voc-size  - max   | mean      | min   | total    | 90th | 95th | 97.5 | 99th
# 2048      - 4567  | 396.69808 | 11    | 9917452  | 794  | 1048 | 1296 | 1579
```
- **Encoding:** A tokenizer with a `VOCAB_SIZE` of 2048 was used to encode the entire training set.
- **Distribution:** The longest review contains **4,567 tokens**, the average is **396**, and the shortest is **11**.
- **Total Volume:** The dataset comprises nearly **10 million** tokens in total.
- **The percentile columns** (90th–99th) are critical for choosing the right `SEQ_LEN` hyperparameter in `config.py`. For example, 
if you set `SEQ_LEN = 1024`, the model's context window will be sufficient for approximately **95 % of all reviews**. 
The remaining 5 % will be truncated according to the `TRUNCATE_END` strategy defined in configuration file.



If you modify the text normalization logic, BPE tokenizer or update the underlying dataset, 
you can generate fresh statistics by running:
```shell
python stats.py
```

## Train the Tokenizer

Before training the neural network, you must train the BPE tokenizer on `datasets/train/`. This creates the vocab and
model files in `assets/`.

```shell
python train_bpe_tokenizer.py
```

*Configuration:* You can adjust `VOCAB_SIZE` and `TEXT_NORMALIZATION_MODE` in `config.py`.

- `TEXT_NORMALIZATION_MODE = standard` - Minimum required text normalization is applied during tokenizer training.
- `TEXT_NORMALIZATION_MODE = aggressive` - Standard normalization plus case and accent insensitive normalization is
  applied during tokenizer training.

## Train the Model

*Configuration:* You can adjust model hyperparameters in `config.py`. `VOCAB_SIZE` and `TEXT_NORMALIZATION_MODE`
must match existing tokenizer.

Run the training script to train the Transformer architecture. The script handles:

- Loading data with dynamic padding/truncation.
- Training with AdamW optimizer.
- Validation at every epoch.
- Saving the best model weights to `weights/`.
- Logging metrics to TensorBoard.

```shell
python train.py
```

To view training progress (Loss, Accuracy, F1-Score, ...):

```shell
tensorboard --logdir runs
```

## Evaluation & Inference

Once training is complete, set the `run_final_test` variable to `True` inside `evaluation.py` and
run the script to get final metrics on the unseen test
set:

```shell
python evaluation.py
```

To make predictions on custom strings run:

```shell
python inference.py
```

## Documentation

The code is divided into separate modules for clarity. There are included one-line comments for comprehension.