"""
Basic train dataset statistics inspection.

This script helps us estimate key hyperparameters such as:
- Sequence length (SEQ_LEN)
- Extent of text normalization
"""
from pathlib import Path

import numpy as np

from dataset import normalize_text
from minbpe.regex import RegexTokenizer

# include only VOCAB_SIZEs of trained tokenizers for both TEXT_NORMALIZATION_MODEs
TRAINED_TOKENIZERS = [512, 1024, 2048, 4096, 8192]


def load_reviews(dataset_dir, polarities):
    reviews = []
    for polarity in polarities:
        # get all .txt files in the train directory
        path = Path(f"{dataset_dir}{polarity}")
        for file_path in path.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                reviews.append(file.read())
    return reviews


def update_char_frequencies(text, char_frequencies):
    # update character frequencies based on a given text
    for char in text:
        if char in char_frequencies:
            char_frequencies[char] += 1
        else:
            char_frequencies[char] = 1


def print_char_stats(title, texts, char_frequencies, top_n=10):
    unique_chars = sorted(set("".join(texts)))

    # print statistics of characters in the provided texts.
    print(f"\n{title}")
    print(f"Unique characters: {unique_chars}")
    print(f"Number of unique characters: {len(unique_chars)}")

    sorted_chars = sorted(char_frequencies.items(), key=lambda item: item[1], reverse=True)

    print(f"\nTop {top_n} most frequent characters:")
    for char, count in sorted_chars[:top_n]:
        print(f"{repr(char)}: {count}")

    print(f"\nTop {top_n} least frequent characters:")
    for char, count in sorted_chars[-top_n:]:
        print(f"{repr(char)}: {count}")


def print_token_stats(vocab_size, reviews_len, mode=""):
    # print tokenization statistics for a set of reviews.
    print("\n- Model vocab size:  ", vocab_size)
    print("- Normalization mode:", mode if mode else "(none)")

    print("Max tokens:          ", reviews_len.max())  # the longest review, measured in tokens
    print("Mean tokens:         ", reviews_len.mean())  # the average review, measured in tokens
    print("Min tokens:          ", reviews_len.min())  # the shortest review, measured in tokens
    print("Total tokens:        ", reviews_len.sum())  # all tokens across the whole dataset

    # calculate percentiles to better estimate seq_len
    for p in [90.0, 95.0, 97.5, 99.0]:
        val = np.percentile(reviews_len, p)
        print(f"{p}th percentile:    {val:.0f}")


def main():
    # initialize tokenizer
    tokenizer = RegexTokenizer()

    # load train reviews (25_000x)
    all_reviews = load_reviews("datasets/train/", ["pos", "neg"])
    print("\nNumber of reviews in train set: ", len(all_reviews))
    standard_norm_reviews = []
    aggressive_norm_reviews = []

    # character frequency
    char_frequencies = {}
    char_freq_standard = {}
    char_freq_aggressive = {}

    for text in all_reviews:
        # update character frequencies for original text
        update_char_frequencies(text, char_frequencies)

        # standard normalization
        standard_norm_text = normalize_text(text, mode="standard")
        standard_norm_reviews.append(standard_norm_text)
        update_char_frequencies(standard_norm_text, char_freq_standard)

        # aggressive normalization
        aggressive_norm_text = normalize_text(text, mode="aggressive")
        aggressive_norm_reviews.append(aggressive_norm_text)
        update_char_frequencies(aggressive_norm_text, char_freq_aggressive)

    print("\n--- Character-level statistics ---")
    print_char_stats("- Original reviews -", all_reviews, char_frequencies)
    print_char_stats("- Normalized reviews - Standard -", standard_norm_reviews, char_freq_standard)
    print_char_stats("- Normalized reviews - Aggressive -", aggressive_norm_reviews, char_freq_aggressive)

    print("\n--- Character-level tokenization statistics ---")
    reviews_len = np.array([len(review) for review in all_reviews])
    print_token_stats("character-level", reviews_len)

    print("\n--- BPE tokenization statistics ---")
    for vocab_size in TRAINED_TOKENIZERS:
        for mode in ["standard", "aggressive"]:
            try:
                tokenizer.load(f"assets/tok{vocab_size}_{mode}.model")
                encoded_reviews = [tokenizer.encode(normalize_text(review, mode=mode)) for review in all_reviews]

                reviews_len = np.array([len(en_review) for en_review in encoded_reviews])

                print_token_stats(vocab_size, reviews_len, mode)
            except FileNotFoundError:
                print(f"[ERROR] Tokenizer for VOCAB_SIZE {vocab_size}, TEXT_NORMALIZATION_MODE {mode} was not found! "
                      f"Train the BPE model first.")


if __name__ == "__main__":
    main()

    # precomputed tokenization statistics for reference

    # BPE tokenization statistics - RAW
    # - Tokenizer was trained on unnormalized text.
    # voc-size  - max   | mean      | min   | total    | 90th | 95th | 97.5 | 99th
    # ch-level  - 13704 | 1325.0696 | 52    | 33126741 | 2617 | 3432 | 4248 | 5213
    # 512       - 6768  | 623.39532 | 19    | 15584883 | 1246 | 1634 | 2025 | 2463
    # 1024      - 5548  | 488.7244  | 14    | 12218110 | 979  | 1288 | 1603 | 1940
    # 2048      - 4806  | 413.4588  | 11    | 10336470 | 828  | 1088 | 1358 | 1649
    # 4096      - 4240  | 363.7482  | 11    | 9093705  | 726  | 957  | 1186 | 1439
    # 8192      - 3851  | 330.18836 | 11    | 8254709  | 657  | 862  | 1069 | 1298

    # BPE tokenization statistics - STANDARD
    # - Minimum required text normalization was applied before tokenizer training.
    # voc-size  - max   | mean      | min   | total    | 90th | 95th | 97.5 | 99th
    # 512       - 6767  | 622.51668 | 19    | 15562917 | 1244 | 1630 | 2016 | 2460
    # 1024      - 5548  | 488.34516 | 14    | 12208629 | 977  | 1286 | 1600 | 1938
    # 2048      - 4806  | 413.22184 | 11    | 10330546 | 828  | 1088 | 1357 | 1648
    # 4096      - 4240  | 363.5408  | 11    | 9088520  | 726  | 956  | 1185 | 1438

    # BPE tokenization statistics - AGGRESSIVE
    # - Standard normalization plus case and accent insensitive normalization was applied before tokenizer training.
    # voc-size  - max   | mean      | min   | total    | 90th | 95th | 97.5 | 99th
    # 512       - 6486  | 596.17904 | 19    | 14904476 | 1193 | 1565 | 1936 | 2362
    # 1024      - 5296  | 468.9656  | 14    | 11724140 | 941  | 1237 | 1534 | 1866
    # 2048      - 4567  | 396.69808 | 11    | 9917452  | 794  | 1048 | 1296 | 1579
    # 4096      - 4026  | 349.62624 | 11    | 8740656  | 699  | 921  | 1140 | 1383
